import os
import json
import uuid
import random
import csv
import io
import sqlite3
import torch
import librosa
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from functools import wraps
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# 导入配置和日志
from config import Config
from utils.logger import logger, log_user_action, log_api_request, log_error, log_model_inference

# 导入情绪安抚pipeline
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline'))
from pipeline.dialogue_pipeline import DialoguePipeline
from vision_model.realtime_detector import VisionEmotionDetector

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 社区图片上传目录
COMMUNITY_IMG_FOLDER = os.path.join('static', 'uploads', 'community')
os.makedirs(COMMUNITY_IMG_FOLDER, exist_ok=True)
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_VOICE_SIZE = Config.VOICE_UPLOAD_MAX_MB * 1024 * 1024
BEIJING_TZ = ZoneInfo("Asia/Shanghai") if ZoneInfo else timezone(timedelta(hours=8))

def beijing_now():
    return datetime.now(BEIJING_TZ)

def beijing_timestamp_str():
    return beijing_now().strftime('%Y-%m-%d %H:%M:%S')

def beijing_date_str():
    return beijing_now().strftime('%Y-%m-%d')

def extract_date_part(value):
    if value is None:
        return ''
    text = str(value).strip()
    if 'T' in text:
        text = text.split('T', 1)[0]
    if ' ' in text:
        text = text.split(' ', 1)[0]
    return text

def build_conversation_title(message):
    if not message:
        return '新对话'
    text = ' '.join(str(message).split())
    for prefix in ['我想', '我觉得', '我有点', '我现在', '我今天', '我最近']:
        if text.startswith(prefix) and len(text) > len(prefix) + 4:
            text = text[len(prefix):].strip()
            break
    text = text.strip('，。！？；：、,.!?;:()（）[]【】 ')
    if len(text) <= 18:
        return text or '新对话'
    return text[:18].rstrip() + '…'

def migrate_existing_timestamps_to_beijing(conn):
    current_version = conn.execute('PRAGMA user_version').fetchone()[0]
    target_version = 1
    if current_version >= target_version:
        return
    timestamp_tables = [
        'users', 'records', 'articles', 'posts', 'post_likes', 'post_comments',
        'dialogue_history', 'books', 'voice_sentences', 'voice_questions', 'voice_records'
    ]
    for table_name in timestamp_tables:
        conn.execute(
            f"""
            UPDATE {table_name}
            SET created_at = datetime(created_at, '+8 hours')
            WHERE created_at IS NOT NULL
              AND created_at GLOB '????-??-?? ??:??:??'
            """
        )
    conn.execute(f'PRAGMA user_version = {target_version}')

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# ── 敏感词检测 ──
_SENSITIVE_WORDS = [
    # 暴力
    '杀人','杀死','去死','自杀','自尽','轻生','跳楼','割腕','烧炭','安眠药过量',
    '绝望死','不想活','活不下去',
    # 色情
    '色情','黄片','裸体','做爱','性交','卖淫','嫖娼','援交',
    # 政治敏感
    '法轮功','天安门事件','推翻政府','颠覆国家',
    # 诈骗/违法
    '洗钱','贩毒','买毒','卖毒','代孕','枪支出售','假证',
    # 侮辱谩骂
    '傻逼','操你','草泥马','狗日','妈的','滚粗','贱人','废物滚',
    '他妈的','fuck you',
]

def detect_sensitive(text: str):
    """检测文本是否包含敏感词，返回 (命中布尔值, 命中词列表)"""
    text_lower = text.lower()
    hits = [w for w in _SENSITIVE_WORDS if w.lower() in text_lower]
    return bool(hits), hits

# 记录应用启动日志
logger.info("心愈 AI 应用启动")
logger.info(f"数据库路径: {Config.DATABASE_PATH}")
logger.info(f"日志级别: {Config.LOG_LEVEL}")
logger.info(f"Dify URL: {Config.DIFY_API_URL}")
logger.info(f"Dify Key 配置状态: {'已配置' if bool(Config.DIFY_API_KEY) else '未配置'}")
logger.info(f"Feishu Webhook 配置状态: {'已配置' if bool(Config.FEISHU_WEBHOOK) else '未配置'}")

# --- 错误处理页面 ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404, error_title="页面未找到", error_message="抱歉，您访问的页面不存在或已被移除。"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_code=500, error_title="服务器错误", error_message="抱歉，服务器遇到了一个问题，请稍后再试。"), 500

@app.errorhandler(403)
def forbidden(e):
    return render_template('error.html', error_code=403, error_title="访问被拒绝", error_message="抱歉，您没有权限访问此页面。"), 403

@app.errorhandler(400)
def bad_request(e):
    return render_template('error.html', error_code=400, error_title="请求无效", error_message="抱歉，您的请求格式有误，请检查后重试。"), 400

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return render_template('error.html', error_code=429, error_title="请求过于频繁", error_message="抱歉，您的请求过于频繁，请稍后再试。"), 429

# --- 全局API错误处理装饰器 ---
@app.after_request
def add_error_handling_headers(response):
    """为API响应添加错误处理支持"""
    # 标记是否为API请求
    if request.path.startswith('/api/'):
        # 如果响应状态码表示错误，添加错误类型标记
        if response.status_code >= 400:
            # 在响应数据中嵌入友好错误信息
            pass
    return response

# --- 认证装饰器 ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- 1. 语音模型核心集成 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Config.MODEL_DIR

class DepressionVoiceModel:
    def __init__(self):
        print("正在初始化语音模型，请稍候...")
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
            self.model.eval()
            print("模型加载完成，运行设备:", DEVICE)
        except Exception as e:
            print(f"模型加载失败，请检查路径或环境: {e}")

    def predict(self, file_path):
        try:
            speech, _ = librosa.load(file_path, sr=16000)
            inputs = self.feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(DEVICE)

            with torch.no_grad():
                logits = self.model(input_values).logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item() 
            confidence = probs[0][prediction].item()
            
            print(f"语音推理完成: Label={prediction}, Confidence={confidence:.2%}")
            return prediction, confidence
        except Exception as e:
            print(f"语音推理出错: {e}")
            return 0, 0.0

# 实例化全局模型对象
voice_analyzer = DepressionVoiceModel()
vision_detector = VisionEmotionDetector()

# --- 2. 数据库工具函数 ---
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    print("[DB] 开始初始化数据库...")
    conn = get_db_connection()
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        # 1. 创建用户表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        
        # 2. 创建问题表 (由 init_db.py 填充数据)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                scale_type TEXT NOT NULL
            )
        ''')

        # 3. 创建记录表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                phq_score INTEGER,
                anxiety_score INTEGER,
                sleep_score INTEGER,
                pressure_score INTEGER,
                social_score INTEGER,
                self_score INTEGER,
                risk_level TEXT,
                voice_label INTEGER,
                voice_confidence REAL,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        
        # 4. 字段检查与修复 (针对旧版本升级)
        cursor = conn.execute('PRAGMA table_info(users)')
        user_columns = [info[1] for info in cursor.fetchall()]
        if 'password' not in user_columns:
            conn.execute('ALTER TABLE users ADD COLUMN password TEXT NOT NULL DEFAULT ""')

        # 5. 检查并初始化题目数据（如果题目表为空）
        cursor = conn.execute('SELECT COUNT(*) FROM questions')
        question_count = cursor.fetchone()[0]
        
        if question_count == 0:
            print("题目库为空，正在初始化题目数据...")
            questions_data = [
                # --- PHQ-9 (抑郁) ---
                ("做事提不起劲头或没有兴趣？", "Depression", "PHQ-9"),
                ("感到心情低落、沮丧或绝望？", "Depression", "PHQ-9"),
                ("入睡困难、睡得不稳或过多？", "Depression", "PHQ-9"),
                ("感觉疲累或没什么精神？", "Depression", "PHQ-9"),
                ("胃口不好或吃得太多？", "Depression", "PHQ-9"),
                ("觉得自己很失败，或让自己及家人失望？", "Depression", "PHQ-9"),
                ("对事物专注有困难，例如看报纸或看电视？", "Depression", "PHQ-9"),
                ("动作或说话速度缓慢到别人已经察觉？", "Depression", "PHQ-9"),
                ("有自伤的想法，或想以某种方式让自己死掉？", "Depression", "PHQ-9"),

                # --- GAD-7 (焦虑) ---
                ("感到紧张、不安或急躁？", "Anxiety", "GAD-7"),
                ("无法停止或控制忧虑？", "Anxiety", "GAD-7"),
                ("对各种各样的事情担忧过多？", "Anxiety", "GAD-7"),
                ("很难放松下来？", "Anxiety", "GAD-7"),
                ("由于不安而无法静坐？", "Anxiety", "GAD-7"),
                ("容易变得烦躁或易怒？", "Anxiety", "GAD-7"),
                ("感到好像有什么可怕的事会发生？", "Anxiety", "GAD-7"),

                # --- Sleep (睡眠质量) ---
                ("觉得入睡困难，躺在床上超过半小时仍很清醒？", "Sleep", "PSS-STYLE"),
                ("在半夜或凌晨惊醒，且难以再次入睡？", "Sleep", "PSS-STYLE"),
                ("因为睡眠质量差而感到白天精神恍惚或疲惫？", "Sleep", "PSS-STYLE"),
                ("需要借助药物、褪黑素等方式辅助入睡？", "Sleep", "PSS-STYLE"),

                # --- Pressure (压力感知) ---
                ("感到无法控制生活中重要的事情？", "Pressure", "PSS-STYLE"),
                ("感到压力大到无法应对？", "Pressure", "PSS-STYLE"),
                ("感到琐事堆积如山，超出了你的处理能力？", "Pressure", "PSS-STYLE"),
                ("很难静下心来放松，总觉得有事情悬而未决？", "Pressure", "PSS-STYLE"),

                # --- Social (社交状态) ---
                ("在社交场合（如聚会、开会）感到局促不安？", "Social", "PSS-STYLE"),
                ("担心别人对自己评价不高或产生误解？", "Social", "PSS-STYLE"),
                ("觉得与周围的人有隔阂，缺乏深层联系？", "Social", "PSS-STYLE"),

                # --- Self (自我价值) ---
                ("对自己正在做的事情失去信心，产生挫败感？", "Self", "PSS-STYLE"),
                ("觉得自己不如周围的人，产生自卑心理？", "Self", "PSS-STYLE"),
                ("觉得自己的努力没有得到应有的认可？", "Self", "PSS-STYLE")
            ]
            conn.executemany(
                'INSERT INTO questions (text, category, scale_type) VALUES (?, ?, ?)', 
                questions_data
            )
            print(f"题目数据初始化完成！共 {len(questions_data)} 道题目。")
        
        # 6. 创建每日打卡表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_checkin (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                mood_score INTEGER,
                mood_label TEXT,
                note TEXT,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')

        # 7. 创建科普文章表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT,
                summary TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        
        # 初始化一些示例文章
        cursor = conn.execute('SELECT COUNT(*) FROM articles')
        article_count = cursor.fetchone()[0]
        
        if article_count == 0:
            sample_articles = [
                ("如何识别抑郁症的早期信号？", "Depression", "了解抑郁症的早期症状可以帮助及时寻求帮助", "抑郁症并非突然发生，它通常有一个渐进的过程。早期信号包括：持续的情绪低落、对以前喜欢的事物失去兴趣、睡眠问题、食欲改变、疲劳感增加、注意力难以集中、自我评价降低等。如果你或身边的人出现这些症状持续两周以上，建议寻求专业心理帮助。"),
                ("焦虑症的常见误解与真相", "Anxiety", "澄清关于焦虑症的错误认识", "关于焦虑症存在许多误解。误解一：焦虑只是\"想太多\"。真相：焦虑症是一种真实的精神障碍，与大脑化学物质失衡有关。误解二：只要\"放松\"就好。真相：虽然放松技巧有帮助，但焦虑症通常需要专业治疗。误解三：药物会让人\"上瘾\"。真相：遵医嘱使用药物是安全有效的治疗方法。"),
                ("改善睡眠质量的10个技巧", "Sleep", "科学有效的睡眠改善方法", "1. 保持规律作息时间\n2. 创造舒适的睡眠环境\n3. 限制咖啡因和酒精摄入\n4. 睡前1小时远离电子设备\n5. 适度运动但避免临睡前剧烈运动\n6. 尝试放松训练如深呼吸\n7. 避免白天长时间小睡\n8. 睡前不要吃太饱\n9. 使用舒适的床上用品\n10. 如果20分钟无法入睡，起来做放松活动再睡"),
                ("压力管理：学会与压力和平共处", "Pressure", "有效管理日常生活中的压力", "压力不一定是坏事，适当的压力可以激励我们前进。但过度的压力会影响身心健康。有效的压力管理包括：1. 识别压力源 2. 学会说\"不\" 3. 时间管理 4. 定期运动 5. 社交支持 6. 正念冥想 7. 寻求专业帮助"),
                ("建立健康社交关系的心理学", "Social", "如何建立和维护良好的人际关系", "健康的人际关系对心理健康至关重要。建立良好社交关系的关键：1. 真诚待人 2. 学会倾听 3. 表达感激 4. 尊重边界 5. 接受不完美 6. 学会独处 7. 处理冲突健康。记住，质量比数量更重要。"),
                ("提升自信心的实用方法", "Self", "科学方法帮助你建立自信", "自信心是可以培养的。以下方法可以帮助你：1. 设定可实现的小目标 2. 记录成功经历 3. 挑战消极想法 4. 练习自我接纳 5. 改善体态 6. 学习新技能 7. 帮助他人。记住，你的价值不取决于他人的评价。")
            ]
            conn.executemany(
                'INSERT INTO articles (title, category, summary, content) VALUES (?, ?, ?, ?)',
                sample_articles
            )
            print(f"文章数据初始化完成！共 {len(sample_articles)} 篇文章。")
        
        # 8. 创建社区帖子表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                anonymous_name TEXT,
                content TEXT,
                mood_tag TEXT,
                image_path TEXT,
                likes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        # 迁移：为旧数据库动态补充 image_path 字段
        existing_cols = [row[1] for row in conn.execute('PRAGMA table_info(posts)').fetchall()]
        if 'image_path' not in existing_cols:
            conn.execute('ALTER TABLE posts ADD COLUMN image_path TEXT')
            print('posts 表已添加 image_path 字段')
        
        # 9. 创建帖子点赞表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS post_likes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours')),
                UNIQUE(post_id, user_id)
            )
        ''')
        
        # 10. 创建帖子评论表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS post_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                user_id INTEGER,
                anonymous_name TEXT,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        
        # 11. 创建对话历史表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS dialogue_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                emotion_label TEXT,
                danger_level TEXT,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        
        # 12. 创建书籍推荐表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT,
                category TEXT,
                description TEXT,
                cover_url TEXT,
                buy_link TEXT,
                is_featured BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        
        # 初始化示例书籍
        cursor = conn.execute('SELECT COUNT(*) FROM books')
        book_count = cursor.fetchone()[0]
        
        if book_count == 0:
            sample_books = [
                ("活出生命的意义", "维克多·弗兰克尔", "自我成长", 
                 "著名心理学家弗兰克尔在纳粹集中营的经历与思考，探讨人类如何在苦难中找到生命的意义。",
                 "https://img.badgen.net/book/9787535487553", "https://example.com/book1", 1),
                ("伯恩斯新情绪疗法", "戴维·伯恩斯", "抑郁", 
                 "一本实用的认知行为疗法指南，帮助读者识别和改变负面思维模式。",
                 "https://img.badgen.net/book/9787559513317", "https://example.com/book2", 1),
                ("正念：此刻是一枝花", "乔·卡巴金", "压力", 
                 "介绍正念减压疗法，通过冥想和觉察减轻压力和焦虑。",
                 "https://img.badgen.net/book/9787508669391", "https://example.com/book3", 0),
                ("我们时代的神经症人格", "卡伦·霍妮", "焦虑", 
                 "经典心理学著作，深入分析现代人的焦虑根源和应对方式。",
                 "https://img.badgen.net/book/9787511740012", "https://example.com/book4", 0),
                ("少有人走的路", "M·斯科特·派克", "自我成长", 
                 "关于心智成熟的旅程，探讨自律、爱和成长的真谛。",
                 "https://img.badgen.net/book/9787111523303", "https://example.com/book5", 1),
                ("情绪的力量", "特拉维斯·斯塔布里克", "情绪管理", 
                 "科学解读情绪的本质，教你如何利用情绪提升生活质量。",
                 "https://img.badgen.net/book/9787521723450", "https://example.com/book6", 0),
            ]
            conn.executemany('''
                INSERT INTO books (title, author, category, description, cover_url, buy_link, is_featured)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', sample_books)
            print(f"书籍数据初始化完成！共 {len(sample_books)} 本书。")
        
        # 13. 创建语音采集句子表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS voice_sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                difficulty TEXT DEFAULT 'medium',
                category TEXT,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        
        # 初始化示例句子
        cursor = conn.execute('SELECT COUNT(*) FROM voice_sentences')
        sentence_count = cursor.fetchone()[0]
        
        if sentence_count == 0:
            sample_sentences = [
                ("今天天气真好，阳光照在身上暖暖的。", "easy", "日常"),
                ("我喜欢在公园里散步，感受大自然的美好。", "easy", "日常"),
                ("周末，我和朋友们一起去看了一场电影，非常开心。", "medium", "日常"),
                ("生活中总会有一些困难，但我们要勇敢面对。", "medium", "励志"),
                ("感谢生命中遇到的每一个人，每一段经历都让我成长。", "medium", "感恩"),
                ("面对挑战时，保持积极的心态是非常重要的。", "hard", "励志"),
                ("我希望通过自己的努力，能够帮助更多的人。", "hard", "励志"),
                ("在忙碌的生活中，我们也要学会停下来，倾听内心的声音。", "hard", "感悟"),
            ]
            conn.executemany('''
                INSERT INTO voice_sentences (content, difficulty, category) VALUES (?, ?, ?)
            ''', sample_sentences)
            print(f"语音采集句子初始化完成！共 {len(sample_sentences)} 句。")
        
        # 14. 创建语音采集问题表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS voice_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                expected_duration INTEGER DEFAULT 30,
                category TEXT,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        
        # 初始化示例问题
        cursor = conn.execute('SELECT COUNT(*) FROM voice_questions')
        question_count = cursor.fetchone()[0]
        
        if question_count == 0:
            sample_questions = [
                ("请描述一下你最近的心情？", 30, "情绪"),
                ("有什么事情让你感到压力很大吗？", 45, "压力"),
                ("你通常是如何放松自己的？", 30, "调节"),
                ("最近有没有什么让你感到开心的事情？", 30, "情绪"),
                ("当你感到焦虑时，你会怎么做？", 45, "调节"),
            ]
            conn.executemany('''
                INSERT INTO voice_questions (content, expected_duration, category) VALUES (?, ?, ?)
            ''', sample_questions)
            print(f"语音采集问题初始化完成！共 {len(sample_questions)} 个问题。")
        
        # 15. 创建语音记录表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS voice_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                module_type TEXT NOT NULL,
                file_path TEXT,
                transcription TEXT,
                emotion_result TEXT,
                duration INTEGER,
                created_at TIMESTAMP DEFAULT (datetime('now', '+8 hours'))
            )
        ''')
        
        # 添加索引以提升查询性能（使用OR IGNORE避免重复创建报错）
        try:
            conn.execute('CREATE INDEX IF NOT EXISTS idx_records_user_id ON records(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_records_created_at ON records(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_checkin_user_id ON daily_checkin(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_checkin_created_at ON daily_checkin(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_post_likes_post_id ON post_likes(post_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_post_comments_post_id ON post_comments(post_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_dialogue_user_id ON dialogue_history(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_dialogue_conversation_id ON dialogue_history(conversation_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_dialogue_user_emotion_label ON dialogue_history(user_id, emotion_label)')
            print("[DB] 数据库索引创建完成")
        except Exception as e:
            print(f"[DB] 索引创建跳过: {e}")

        migrate_existing_timestamps_to_beijing(conn)

        conn.commit()
    finally:
        conn.close()

# --- 简单的健康检查端点 ---
@app.route('/api/health')
def health_check():
    return jsonify({"status": "ok", "time": beijing_now().isoformat()})

# 在启动前初始化
init_db()

# --- 3. 页面路由 ---
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            # 登录成功，跳转到首页并传递登录成功参数
            return redirect(url_for('index', login_success='true'))
        flash('用户名或密码错误')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('两次输入的密码不一致')
            return render_template('register.html')
        
        # 输入验证
        if not username or len(username) < 3 or len(username) > 20:
            flash('用户名需要3-20个字符')
            return render_template('register.html')
        if not password or len(password) < 6:
            flash('密码至少需要6个字符')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)', (username, hashed_password, beijing_timestamp_str()))
            conn.commit()
            log_user_action('register', username=username)
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('用户名已存在')
        except Exception as e:
            log_error(e, context=f"register|username={username}")
            flash('注册失败，请稍后再试')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/detect')
@login_required
def detect(): return render_template('detect.html')

@app.route('/report')
@login_required
def report(): return render_template('report.html')

@app.route('/soothe')
def soothe(): return render_template('soothe.html')

@app.route('/profile')
def profile(): 
    # 允许匿名访问，但未登录时显示提示
    user_id = session.get('user_id')
    username = session.get('username', '匿名用户')
    return render_template('profile.html', is_anonymous=(user_id is None), username=username)

# --- 文章库页面 ---
@app.route('/articles')
def articles():
    return render_template('articles.html')

@app.route('/article/<int:article_id>')
def article_detail(article_id):
    conn = get_db_connection()
    article = conn.execute('SELECT * FROM articles WHERE id = ?', (article_id,)).fetchone()
    conn.close()
    if article:
        return render_template('article_detail.html', article=dict(article))
    return "文章不存在", 404

# --- 文章 API ---
@app.route('/api/articles')
def get_articles():
    category = request.args.get('category')
    conn = get_db_connection()
    if category:
        articles = conn.execute('SELECT id, title, category, summary FROM articles WHERE category = ?', (category,)).fetchall()
    else:
        articles = conn.execute('SELECT id, title, category, summary FROM articles').fetchall()
    conn.close()
    return jsonify([dict(a) for a in articles])

@app.route('/api/article/<int:article_id>')
def get_article(article_id):
    conn = get_db_connection()
    article = conn.execute('SELECT * FROM articles WHERE id = ?', (article_id,)).fetchone()
    conn.close()
    if article:
        return jsonify(dict(article))
    return jsonify({"error": "文章不存在"}), 404

@app.route('/api/articles/recommend')
def get_recommended_articles():
    """
    根据用户测评结果推荐相关文章
    category: 主要问题类别（Depression, Anxiety, Sleep, Pressure, Social, Self）
    """
    category = request.args.get('category', '')
    limit = int(request.args.get('limit', 3))

    conn = get_db_connection()

    # 如果有指定分类，优先返回该分类的文章
    if category:
        articles = conn.execute(
            'SELECT id, title, category, summary FROM articles WHERE category = ? LIMIT ?',
            (category, limit)
        ).fetchall()

        # 如果该分类文章不足3篇，补充其他分类
        if len(articles) < limit:
            others = conn.execute(
                'SELECT id, title, category, summary FROM articles WHERE category != ? LIMIT ?',
                (category, limit - len(articles))
            ).fetchall()
            articles = list(articles) + list(others)
    else:
        # 无分类时返回最新文章
        articles = conn.execute(
            'SELECT id, title, category, summary FROM articles ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()

    conn.close()
    return jsonify([dict(a) for a in articles])

# --- 书籍推荐页面 ---
@app.route('/books')
def books():
    """书籍推荐页面"""
    return render_template('books.html')

@app.route('/api/books')
def get_books():
    """获取书籍列表"""
    category = request.args.get('category')
    featured = request.args.get('featured')
    
    conn = get_db_connection()
    
    query = 'SELECT * FROM books WHERE 1=1'
    params = []
    
    if category:
        query += ' AND category = ?'
        params.append(category)
    
    if featured == 'true':
        query += ' AND is_featured = 1'
    
    query += ' ORDER BY is_featured DESC, created_at DESC'
    
    books = conn.execute(query, params).fetchall()
    conn.close()
    
    return jsonify([dict(b) for b in books])

@app.route('/api/books/categories')
def get_book_categories():
    """获取书籍分类列表"""
    conn = get_db_connection()
    categories = conn.execute('SELECT DISTINCT category FROM books WHERE category IS NOT NULL').fetchall()
    conn.close()
    return jsonify([c['category'] for c in categories])

# --- 语音采集页面 ---
@app.route('/voice')
def voice_collect():
    """语音采集已整合到测评流程，重定向到检测页"""
    return redirect(url_for('detect'))

@app.route('/api/voice/sentences')
def get_voice_sentences():
    """获取随机朗读句子"""
    count = int(request.args.get('count', 5))
    
    conn = get_db_connection()
    sentences = conn.execute('''
        SELECT * FROM voice_sentences 
        ORDER BY RANDOM() 
        LIMIT ?
    ''', (count,)).fetchall()
    conn.close()
    
    return jsonify([dict(s) for s in sentences])

@app.route('/api/voice/questions')
def get_voice_questions():
    """获取问答问题"""
    conn = get_db_connection()
    questions = conn.execute('SELECT * FROM voice_questions ORDER BY id').fetchall()
    conn.close()
    return jsonify([dict(q) for q in questions])

@app.route('/api/voice/upload', methods=['POST'])
def upload_voice():
    """上传录音文件"""
    try:
        user_id = session.get('user_id')
        
        if 'file' not in request.files:
            return jsonify({"error": "没有文件"}), 400
        
        file = request.files['file']
        module_type = request.form.get('module_type', 'reading')
        duration = int(request.form.get('duration', 0))
        
        if file.filename == '':
            return jsonify({"error": "文件名为空"}), 400

        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in {'webm', 'wav', 'mp3', 'm4a', 'ogg'}:
            return jsonify({"error": "仅支持 webm/wav/mp3/m4a/ogg 音频格式"}), 400

        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        if size > MAX_VOICE_SIZE:
            return jsonify({"error": f"音频大小不能超过 {Config.VOICE_UPLOAD_MAX_MB}MB"}), 400

        # 保存文件
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 简单模拟情绪分析结果
        emotion_result = {
            "emotion": "平静",
            "confidence": 0.85,
            "stress_level": "low"
        }
        
        # 保存记录到数据库
        conn = get_db_connection()
        cursor = conn.execute('''
            INSERT INTO voice_records (user_id, module_type, file_path, duration, emotion_result, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, module_type, filepath, duration, json.dumps(emotion_result), beijing_timestamp_str()))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        log_user_action('voice_upload', username=session.get('username'), details=f"type={module_type}")
        
        return jsonify({
            "success": True,
            "record_id": record_id,
            "result": emotion_result
        })
    except Exception as e:
        log_error(e, context="upload_voice")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/result', methods=['POST'])
def get_voice_result():
    """综合语音和量表结果"""
    try:
        data = request.get_json()
        scale_score = data.get('scale_score', 0)  # 量表分数 0-27
        voice_results = data.get('voice_results', [])
        
        # 融合算法：量表 60% + 语音 40%
        voice_avg = 0
        if voice_results:
            # 简单计算平均情绪分数
            voice_avg = sum([r.get('confidence', 0.5) for r in voice_results]) / len(voice_results)
        
        # 将语音置信度转换为抑郁分数（反向，置信度高表示情绪正常）
        voice_score = (1 - voice_avg) * 27  # 转换为 0-27 分数
        
        # 综合分数
        final_score = int(scale_score * 0.6 + voice_score * 0.4)
        
        # 风险评估
        if final_score < 5:
            risk_level = "低"
            risk_color = "green"
        elif final_score < 10:
            risk_level = "中"
            risk_color = "yellow"
        else:
            risk_level = "高"
            risk_color = "red"
        
        return jsonify({
            "final_score": final_score,
            "scale_score": scale_score,
            "voice_score": int(voice_score),
            "risk_level": risk_level,
            "risk_color": risk_color
        })
    except Exception as e:
        log_error(e, context="get_voice_result")
        return jsonify({"error": str(e)}), 500

# --- 社区页面 ---
@app.route('/community')
def community():
    return render_template('community.html')

# --- 社区图片上传 API ---
@app.route('/api/posts/upload-image', methods=['POST'])
def upload_community_image():
    """上传社区帖子图片"""
    if 'user_id' not in session:
        return jsonify({'error': '请先登录'}), 401
    if 'image' not in request.files:
        return jsonify({'error': '未选择图片'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择图片'}), 400
    if not allowed_image(file.filename):
        return jsonify({'error': '仅支持 PNG、JPG、GIF、WEBP 格式'}), 400
    # 检查文件大小
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_IMAGE_SIZE:
        return jsonify({'error': '图片大小不能超过 5MB'}), 400
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(COMMUNITY_IMG_FOLDER, filename)
    file.save(save_path)
    url = f"/static/uploads/community/{filename}"
    return jsonify({'success': True, 'url': url})

# --- 社区 API ---
@app.route('/api/posts', methods=['GET', 'POST'])
def get_posts():
    if request.method == 'POST':
        # 发布帖子
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "请先登录"}), 401
        
        try:
            data = request.get_json()
            content = data.get('content', '').strip()
            mood_tag = data.get('mood_tag', '')
            image_path = data.get('image_path', '')

            if not content:
                return jsonify({"error": "内容不能为空"}), 400
            if len(content) > 1000:
                return jsonify({"error": "内容不能超过1000字"}), 400

            # 敏感词检测
            hit, words = detect_sensitive(content)
            if hit:
                return jsonify({"error": "内容含有违规词汇，请修改后发布", "hit_count": len(words)}), 400

            # 生成匿名昵称
            anonymous_names = ['倾听者', '阳光小伙伴', '心灵旅者', '温暖的人', '努力的人', '勇敢的心', '微笑面对', '宁静致远', '追光者', '破晓']
            anonymous_name = random.choice(anonymous_names) + str(random.randint(1, 999))

            # 校验 image_path 只能是本站上传路径
            if image_path and not image_path.startswith('/static/uploads/community/'):
                return jsonify({"error": "图片路径非法"}), 400
            
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO posts (user_id, anonymous_name, content, mood_tag, image_path, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                (user_id, anonymous_name, content, mood_tag, image_path, beijing_timestamp_str())
            )
            conn.commit()
            conn.close()
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    else:
        # 获取帖子列表（支持分页）
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 10))
        offset = (page - 1) * page_size

        conn = get_db_connection()

        # 获取总数
        total_count = conn.execute('SELECT COUNT(*) FROM posts').fetchone()[0]

        # 获取当前页数据
        posts = conn.execute('''
            SELECT p.*,
                   (SELECT COUNT(*) FROM post_likes WHERE post_id = p.id) as like_count,
                   (SELECT COUNT(*) FROM post_comments WHERE post_id = p.id) as comment_count,
                   CASE WHEN ? > 0 AND (SELECT COUNT(*) FROM post_likes WHERE post_id = p.id AND user_id = ?) > 0 THEN 1 ELSE 0 END as liked
            FROM posts p
            ORDER BY p.created_at DESC
            LIMIT ? OFFSET ?
        ''', (session.get('user_id', 0), session.get('user_id', 0), page_size, offset)).fetchall()

        conn.close()

        # 返回分页数据和元信息
        return jsonify({
            'posts': [dict(p) for p in posts],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': (total_count + page_size - 1) // page_size,
                'has_next': page * page_size < total_count,
                'has_prev': page > 1
            }
        })

@app.route('/api/posts/<int:post_id>/like', methods=['POST'])
def like_post(post_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "请先登录"}), 401
    
    try:
        conn = get_db_connection()
        # 检查是否已点赞
        existing = conn.execute('SELECT * FROM post_likes WHERE post_id = ? AND user_id = ?', 
                                (post_id, user_id)).fetchone()
        
        if existing:
            # 取消点赞
            conn.execute('DELETE FROM post_likes WHERE post_id = ? AND user_id = ?', (post_id, user_id))
            conn.execute('UPDATE posts SET likes = likes - 1 WHERE id = ?', (post_id,))
            liked = False
        else:
            # 添加点赞
            conn.execute('INSERT INTO post_likes (post_id, user_id, created_at) VALUES (?, ?, ?)', (post_id, user_id, beijing_timestamp_str()))
            conn.execute('UPDATE posts SET likes = likes + 1 WHERE id = ?', (post_id,))
            liked = True
        
        conn.commit()
        new_likes = conn.execute('SELECT likes FROM posts WHERE id = ?', (post_id,)).fetchone()['likes']
        conn.close()
        
        return jsonify({"success": True, "liked": liked, "likes": new_likes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 社区评论 API ---
@app.route('/api/posts/<int:post_id>/comments', methods=['GET'])
def get_comments(post_id):
    """获取帖子的评论列表"""
    try:
        conn = get_db_connection()
        comments = conn.execute('''
            SELECT c.*, 
                   (SELECT COUNT(*) FROM post_comments WHERE post_id = c.post_id) as comment_count
            FROM post_comments c
            WHERE c.post_id = ?
            ORDER BY c.created_at ASC
        ''', (post_id,)).fetchall()
        conn.close()
        return jsonify([dict(c) for c in comments])
    except Exception as e:
        log_error(e, context=f"get_comments|post_id={post_id}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/posts/<int:post_id>/comments', methods=['POST'])
def add_comment(post_id):
    """添加评论"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "请先登录"}), 401
        
        data = request.get_json()
        content = data.get('content', '').strip()
        
        if not content:
            return jsonify({"error": "评论内容不能为空"}), 400
        
        if len(content) > 500:
            return jsonify({"error": "评论内容不能超过500字"}), 400

        # 敏感词检测
        hit, words = detect_sensitive(content)
        if hit:
            return jsonify({"error": "评论含有违规词汇，请修改后发布", "hit_count": len(words)}), 400
        
        user_id = session['user_id']
        username = session.get('username', '匿名用户')
        
        conn = get_db_connection()
        # 检查帖子是否存在
        post = conn.execute('SELECT id FROM posts WHERE id = ?', (post_id,)).fetchone()
        if not post:
            conn.close()
            return jsonify({"error": "帖子不存在"}), 404
        
        cursor = conn.execute('''
            INSERT INTO post_comments (post_id, user_id, anonymous_name, content, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (post_id, user_id, username, content, beijing_timestamp_str()))
        conn.commit()
        
        # 获取刚插入的评论
        comment = conn.execute('SELECT * FROM post_comments WHERE id = ?', (cursor.lastrowid,)).fetchone()
        conn.close()
        
        log_user_action('add_comment', username=username, details=f"post_id={post_id}")
        return jsonify({"success": True, "comment": dict(comment)})
    except Exception as e:
        log_error(e, context="add_comment")
        return jsonify({"error": str(e)}), 500

@app.route('/api/comments/<int:comment_id>', methods=['DELETE'])
def delete_comment(comment_id):
    """删除评论"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "请先登录"}), 401
        
        user_id = session['user_id']
        
        conn = get_db_connection()
        # 检查评论是否存在且属于当前用户
        comment = conn.execute('SELECT * FROM post_comments WHERE id = ?', (comment_id,)).fetchone()
        
        if not comment:
            conn.close()
            return jsonify({"error": "评论不存在"}), 404
        
        if comment['user_id'] != user_id:
            conn.close()
            return jsonify({"error": "无权删除此评论"}), 403
        
        conn.execute('DELETE FROM post_comments WHERE id = ?', (comment_id,))
        conn.commit()
        conn.close()
        
        log_user_action('delete_comment', username=session.get('username'), details=f"comment_id={comment_id}")
        return jsonify({"success": True})
    except Exception as e:
        log_error(e, context=f"delete_comment|comment_id={comment_id}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/posts/<int:post_id>', methods=['DELETE'])
def delete_post(post_id):
    """删除帖子（仅作者本人可删除）"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "请先登录"}), 401

        user_id = session['user_id']
        conn = get_db_connection()
        post = conn.execute('SELECT * FROM posts WHERE id = ?', (post_id,)).fetchone()

        if not post:
            conn.close()
            return jsonify({"error": "帖子不存在"}), 404

        if post['user_id'] != user_id:
            conn.close()
            return jsonify({"error": "无权删除此帖子"}), 403

        # 删除帖子及其关联的点赞和评论
        conn.execute('DELETE FROM post_comments WHERE post_id = ?', (post_id,))
        conn.execute('DELETE FROM post_likes WHERE post_id = ?', (post_id,))
        conn.execute('DELETE FROM posts WHERE id = ?', (post_id,))
        conn.commit()

        # 若有图片，删除图片文件（提前保存路径，close 后再操作文件系统）
        image_path = post['image_path']
        conn.close()

        if image_path and image_path.startswith('/static/uploads/community/'):
            rel = image_path.lstrip('/').replace('/', os.sep)
            img_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
            try:
                if os.path.exists(img_file):
                    os.remove(img_file)
            except OSError as file_err:
                log_error(file_err, context=f"delete_post_image|path={img_file}")

        log_user_action('delete_post', username=session.get('username'), details=f"post_id={post_id}")
        return jsonify({"success": True})
    except Exception as e:
        log_error(e, context=f"delete_post|post_id={post_id}")
        return jsonify({"error": str(e)}), 500

# --- 4. 核心 API 接口 ---
@app.route('/api/questions')
def get_questions():
    conn = get_db_connection()
    questions = conn.execute('SELECT * FROM questions').fetchall()
    conn.close()
    return jsonify([dict(q) for q in questions])

@app.route('/api/vision/analyze', methods=['POST'])
def analyze_vision_frame():
    """实时视觉情绪检测（接收前端摄像头帧）"""
    try:
        data = request.get_json() or {}
        if data.get('reset'):
            vision_detector.reset_smoothing()
            return jsonify({"ready": True, "reset": True})

        frame = data.get('frame', '')
        result = vision_detector.analyze_frame(frame)
        return jsonify(result)
    except Exception as e:
        log_error(e, context='analyze_vision_frame')
        return jsonify({
            "emotion": "Neutral",
            "confidence": 0.0,
            "depression_score": 0.0,
            "smoothed_score": 0.0,
            "ready": False,
            "error": str(e)
        }), 200

@app.route('/api/submit', methods=['POST'])
def submit_assessment():
    try:
        user_answers = json.loads(request.form.get('answers', '{}'))
        audio_file = request.files.get('audio')
        
        # 获取当前用户ID（如果已登录）
        user_id = session.get('user_id')
        
        v_label = 0
        v_conf = 0.0
        visual_score = float(request.form.get('vision_score', 0) or 0)
        visual_emotion = request.form.get('vision_emotion', 'Neutral')

        if audio_file:
            filename = f"voice_{uuid.uuid4().hex}.wav"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            audio_file.save(file_path)
            v_label, v_conf = voice_analyzer.predict(file_path)

        # 扩展所有新维度
        scores = {
            "Depression": 0, 
            "Anxiety": 0, 
            "Sleep": 0, 
            "Pressure": 0,
            "Social": 0,
            "Self": 0
        }
        
        conn = get_db_connection()
        for q_id, val in user_answers.items():
            q = conn.execute('SELECT category FROM questions WHERE id=?', (q_id,)).fetchone()
            if q and q['category'] in scores:
                scores[q['category']] += int(val)
        
        # --- 综合风险判定逻辑（量表 + 语音 + 视觉） ---
        phq = scores["Depression"]
        if phq >= 15 or v_label == 1 or visual_score >= 0.65:
            risk = "高风险"
        elif phq >= 7 or visual_score >= 0.45:
            risk = "中等风险"
        else:
            risk = "低风险"

        # 存入数据库（包含 user_id 以关联用户）
        conn.execute('''
            INSERT INTO records (
                user_id, phq_score, anxiety_score, sleep_score, pressure_score, 
                social_score, self_score, risk_level, voice_label, voice_confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            scores["Depression"], 
            scores["Anxiety"], 
            scores["Sleep"], 
            scores["Pressure"],
            scores["Social"],
            scores["Self"],
            risk, 
            v_label,
            v_conf,
            beijing_timestamp_str()
        ))
        conn.commit()
        conn.close()

        # --- 雷达图归一化处理 (折算为10分制) ---
        # 计算公式：(实际得分 / 该维度最大可能得分) * 10
        voice_radar = round((v_label * 7 + 2), 1)
        visual_radar = round(max(0.0, min(10.0, visual_score * 10)), 1)
        multimodal_radar = round(max(voice_radar, visual_radar), 1)

        radar_data = [
            round((scores["Depression"] / 27) * 10, 1), # PHQ-9 满分 27
            round((scores["Anxiety"] / 21) * 10, 1),    # GAD-7 满分 21
            round((scores["Sleep"] / 12) * 10, 1),      # 4题 满分 12
            round((scores["Pressure"] / 12) * 10, 1),   # 4题 满分 12
            round((scores["Social"] / 9) * 10, 1),      # 3题 满分 9
            round((scores["Self"] / 9) * 10, 1),        # 3题 满分 9
            multimodal_radar                              # 多模态风险映射（语音+视觉）
        ]

        # 计算主要问题类别（得分最高的维度）
        category_scores = {
            "Depression": scores["Depression"],
            "Anxiety": scores["Anxiety"],
            "Sleep": scores["Sleep"],
            "Pressure": scores["Pressure"],
            "Social": scores["Social"],
            "Self": scores["Self"]
        }
        highest_category = max(category_scores, key=category_scores.get)

        return jsonify({
            "risk_level": risk,
            "radar_data": radar_data,
            "scores": scores, # 将原始分数也传回前端，方便详细显示
            "highest_category": highest_category,  # 主要问题类别
            "voice": {
                "label": int(v_label),
                "confidence": float(v_conf)
            },
            "vision": {
                "emotion": visual_emotion,
                "score": float(visual_score)
            }
        })
    except Exception as e:
        print(f"Submit Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- 增加获取历史记录接口 ---
@app.route('/api/history')
def get_history():
    try:
        user_id = session.get('user_id')
        conn = get_db_connection()

        # 如果已登录，只获取当前用户的历史记录
        if user_id:
            records = conn.execute(
                'SELECT * FROM records WHERE user_id = ? ORDER BY created_at DESC LIMIT 100',
                (user_id,)
            ).fetchall()
        else:
            # 未登录时返回空列表
            records = []

        conn.close()
        return jsonify([dict(r) for r in records])
    except Exception as e:
        print(f"History Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- 每日打卡 API ---
@app.route('/api/checkin', methods=['GET', 'POST'])
def checkin():
    if request.method == 'GET':
        # 获取当前用户的打卡记录
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "请先登录"}), 401
        try:
            conn = get_db_connection()
            records = conn.execute(
                'SELECT * FROM daily_checkin WHERE user_id = ? ORDER BY created_at DESC LIMIT 30',
                (user_id,)
            ).fetchall()
            conn.close()
            return jsonify([dict(r) for r in records])
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    else:  # POST - 提交打卡
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "请先登录"}), 401
        try:
            data = request.get_json()
            mood_score = data.get('mood_score', 0)
            mood_label = data.get('mood_label', '')
            note = data.get('note', '')
            
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO daily_checkin (user_id, mood_score, mood_label, note, created_at) VALUES (?, ?, ?, ?, ?)',
                (user_id, mood_score, mood_label, note, beijing_timestamp_str())
            )
            conn.commit()
            conn.close()
            return jsonify({"success": True, "message": "打卡成功"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# --- 打卡数据导出 API ---
@app.route('/api/checkin/export')
def export_checkin():
    """导出用户的打卡数据为 CSV 格式"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "请先登录"}), 401
        
        conn = get_db_connection()
        records = conn.execute('''
            SELECT created_at as 日期, mood_score as 情绪分数, mood_label as 情绪标签, note as 笔记
            FROM daily_checkin
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,)).fetchall()
        conn.close()
        
        if not records:
            return jsonify({"error": "暂无打卡记录"}), 404
        
        # 生成 CSV 内容
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入表头
        writer.writerow(['日期', '情绪分数', '情绪标签', '笔记'])
        
        # 写入数据
        for record in records:
            writer.writerow([
                record['日期'],
                record['情绪分数'],
                record['情绪标签'],
                record['笔记'] or ''
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        # 返回 CSV 文件
        response = Response(
            csv_content,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=checkin_export_{beijing_now().strftime("%Y%m%d")}.csv'
            }
        )
        return response
    except Exception as e:
        log_error(e, context="export_checkin")
        return jsonify({"error": str(e)}), 500

# --- 获取用户统计数据 API ---
@app.route('/api/stats')
def get_stats():
    print("[API] /api/stats 被调用")
    user_id = session.get('user_id')
    print(f"[API] user_id = {user_id}")

    if not user_id:
        print("[API] 用户未登录，返回401")
        return jsonify({"error": "请先登录"}), 401

    try:
        print(f"[Stats] 正在获取用户 {user_id} 的统计数据...")

        # 设置更短的超时
        conn = get_db_connection()
        conn.execute("PRAGMA query_only = OFF")
        conn.execute("PRAGMA busy_timeout = 5000")

        # 1. 获取测评记录统计（限制最近200条）
        records = conn.execute(
            'SELECT * FROM records WHERE user_id = ? ORDER BY created_at DESC LIMIT 200',
            (user_id,)
        ).fetchall()
        print(f"[Stats] 测评记录数: {len(records)}")

        # 2. 获取打卡记录（限制最近100条）
        checkins = conn.execute(
            'SELECT * FROM daily_checkin WHERE user_id = ? ORDER BY created_at DESC LIMIT 100',
            (user_id,)
        ).fetchall()
        print(f"[Stats] 打卡记录数: {len(checkins)}")
        
        # 3. 计算统计数据
        total_assessments = len(records)
        total_checkins = len(checkins)
        
        # 计算连续打卡天数
        consecutive_days = 0
        if checkins:
            # 提取日期部分（只取日期，不要时间），统一格式
            checkin_dates = set()
            for c in checkins:
                created_at = c['created_at']
                # 如果是完整时间格式，提取日期部分
                if isinstance(created_at, str) and ' ' in created_at:
                    checkin_dates.add(extract_date_part(created_at))
                else:
                    checkin_dates.add(extract_date_part(created_at))
            
            today = beijing_now().date()
            for i in range(30):
                check_date = (today - timedelta(days=i)).isoformat()
                if check_date in checkin_dates:
                    consecutive_days += 1
                else:
                    break
        
        # 计算平均分数
        avg_phq = 0
        avg_anxiety = 0
        if records:
            avg_phq = sum(r['phq_score'] for r in records) / len(records)
            avg_anxiety = sum(r['anxiety_score'] for r in records) / len(records)
        
        # 心理健康评分 (0-100)
        mental_score = max(0, 100 - (avg_phq / 27 * 50) - (avg_anxiety / 21 * 30) - (20 if total_checkins == 0 else 0))
        
        # 4. 获取最近7天的打卡趋势
        recent_checkins = []
        for i in range(6, -1, -1):
            date = (beijing_now() - timedelta(days=i)).date().isoformat()
            # 统一日期格式进行比较
            checkin = None
            for c in checkins:
                checkin_date = extract_date_part(c['created_at'])
                if checkin_date == date:
                    checkin = c
                    break
            recent_checkins.append({
                'date': date,
                'mood': checkin['mood_score'] if checkin else None
            })
        
        conn.close()
        
        return jsonify({
            'total_assessments': total_assessments,
            'total_checkins': total_checkins,
            'consecutive_days': consecutive_days,
            'avg_phq': round(avg_phq, 1),
            'avg_anxiety': round(avg_anxiety, 1),
            'mental_score': round(mental_score),
            'recent_checkins': recent_checkins,
            'risk_distribution': {
                'low': len([r for r in records if r['risk_level'] == '低']),
                'medium': len([r for r in records if r['risk_level'] == '中']),
                'high': len([r for r in records if r['risk_level'] == '高'])
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 数据导出 API ---
@app.route('/api/export')
def export_data():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "请先登录"}), 401
    
    try:
        conn = get_db_connection()
        
        # 获取评估记录
        records = conn.execute(
            'SELECT * FROM records WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        ).fetchall()
        
        # 获取打卡记录
        checkins = conn.execute(
            'SELECT * FROM daily_checkin WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        ).fetchall()
        conn.close()
        
        # 生成 CSV
        output = io.StringIO()
        
        # 评估记录
        output.write('=== 心理健康评估记录 ===\n')
        output.write('ID,抑郁指数,焦虑指数,睡眠指数,压力指数,社交指数,自我价值指数,风险等级,语音标签,置信度,创建时间\n')
        for r in records:
            output.write(f"{r['id']},{r['phq_score']},{r['anxiety_score']},{r['sleep_score']},{r['pressure_score']},{r['social_score']},{r['self_score']},{r['risk_level']},{r['voice_label']},{r['voice_confidence']},{r['created_at']}\n")
        
        output.write('\n=== 每日情绪打卡记录 ===\n')
        output.write('ID,情绪评分,情绪标签,备注,日期\n')
        for c in checkins:
            output.write(f"{c['id']},{c['mood_score']},{c['mood_label']},{c['note']},{c['created_at']}\n")
        
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=心理健康数据_{user_id}.csv'
        
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 历史记录页面 ---
@app.route('/history')
def history():
    return render_template('history.html')

# --- 个人资料 API ---
@app.route('/api/profile')
def get_profile():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'total_detects': 0, 'consecutive_days': 0, 'mental_health_score': 0})
    try:
        conn = get_db_connection()
        records = conn.execute('SELECT * FROM records WHERE user_id = ? ORDER BY created_at DESC LIMIT 100', (user_id,)).fetchall()
        checkins = conn.execute('SELECT * FROM daily_checkin WHERE user_id = ? ORDER BY created_at DESC LIMIT 30', (user_id,)).fetchall()
        conn.close()
        total_detects = len(records)
        consecutive_days = 0
        if checkins:
            checkin_dates = set()
            for c in checkins:
                created_at = c['created_at']
                checkin_dates.add(extract_date_part(created_at))
            today = beijing_now().date()
            for i in range(30):
                if (today - timedelta(days=i)).isoformat() in checkin_dates:
                    consecutive_days += 1
                else:
                    break
        avg_phq = sum(r['phq_score'] for r in records) / len(records) if records else 0
        avg_anxiety = sum(r['anxiety_score'] for r in records) / len(records) if records else 0
        mental_score = max(0, round(100 - (avg_phq / 27 * 50) - (avg_anxiety / 21 * 30)))
        return jsonify({'total_detects': total_detects, 'consecutive_days': consecutive_days, 'mental_health_score': mental_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 今日打卡状态 ---
@app.route('/api/checkin/today')
def checkin_today():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'checked_in': False})
    try:
        conn = get_db_connection()
        today = beijing_date_str()
        record = conn.execute(
            "SELECT * FROM daily_checkin WHERE user_id = ? AND (created_at = ? OR created_at LIKE ?) ORDER BY id DESC LIMIT 1",
            (user_id, today, today + '%')
        ).fetchone()
        conn.close()
        if record:
            return jsonify({'checked_in': True, 'mood_score': record['mood_score'], 'mood_label': record['mood_label']})
        return jsonify({'checked_in': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 近7天打卡趋势 ---
@app.route('/api/checkin/week')
def checkin_week():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify([])
    try:
        conn = get_db_connection()
        checkins = conn.execute(
            'SELECT * FROM daily_checkin WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT 30',
            (user_id,)
        ).fetchall()
        conn.close()
        result = []
        for i in range(6, -1, -1):
            date = (beijing_now() - timedelta(days=i)).date().isoformat()
            mood = None
            for c in checkins:
                date_part = extract_date_part(c['created_at'])
                if date_part == date:
                    mood = c['mood_score']
                    break
            result.append({'date': date[-5:], 'mood_score': mood or 0})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# 全局pipeline实例（延迟初始化）
_dialogue_pipeline = None

def get_dialogue_pipeline():
    global _dialogue_pipeline
    if _dialogue_pipeline is None:
        _dialogue_pipeline = DialoguePipeline()
    return _dialogue_pipeline


def _extract_dify_error_message(response):
    default_msg = '智能体暂时不可用，请稍后重试'
    try:
        err_data = response.json()
        return err_data.get('message') or err_data.get('error') or default_msg
    except Exception:
        return default_msg


def _is_conversation_not_exists(error_msg):
    msg = (error_msg or '').lower()
    return 'conversation not exists' in msg or 'conversation_not_exists' in msg

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    情绪垃圾桶聊天接口
    接收用户文本，调用dialogue_pipeline进行情绪分析并生成安抚回复
    支持对话上下文记忆（通过 session 保存历史）
    """
    try:
        data = request.get_json() or {}
        user_text = data.get('text', '').strip()
        conversation_id = data.get('conversation_id', 'default')
        
        if not user_text:
            return jsonify({'reply': '我在这里倾听你，请告诉我你的想法。'}), 400
        if len(user_text) > 1000:
            return jsonify({'reply': '输入内容过长，请控制在1000字以内。'}), 400
        
        # 获取对话历史（从 session 中）
        chat_history = session.get('chat_history', [])
        
        # 调用情绪安抚pipeline，传入历史对话
        pipeline = get_dialogue_pipeline()
        try:
            result, emotion_result = pipeline.run(user_text, history=chat_history)
            reply = result if isinstance(result, str) else str(result)
            emotion_label = emotion_result.get('emotion', '平静')
            # 将 risk_level 映射为 danger_level
            risk_map = {'emergency': '极高', '高': '高', '中': '中', '低': '低'}
            danger_level = risk_map.get(emotion_result.get('risk_level', '低'), '低')
        except Exception as pipeline_err:
            logger.error(f"Pipeline运行异常: {pipeline_err}")
            reply = '我在这里陪伴你，如果有任何不适，请寻求专业帮助。'
            emotion_label = '平静'
            danger_level = '低'
        
        # 更新对话历史
        chat_history.append({'role': 'user', 'content': user_text})
        chat_history.append({'role': 'assistant', 'content': reply})
        
        # 只保留最近10轮对话（20条记录）
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        session['chat_history'] = chat_history
        
        # 保存到数据库
        user_id = session.get('user_id')
        if user_id:
            try:
                conn = get_db_connection()
                # 保存用户消息
                conn.execute('''
                    INSERT INTO dialogue_history (user_id, conversation_id, role, message, emotion_label, danger_level, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, conversation_id, 'user', user_text, emotion_label, danger_level, beijing_timestamp_str()))
                # 保存AI回复（AI消息不做情绪标注）
                conn.execute('''
                    INSERT INTO dialogue_history (user_id, conversation_id, role, message, emotion_label, danger_level, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, conversation_id, 'assistant', reply, None, None, beijing_timestamp_str()))
                conn.commit()
                conn.close()
            except Exception as db_error:
                print(f"保存对话历史失败: {db_error}")
        
        return jsonify({
            'reply': reply,
            'emotion_label': emotion_label,
            'danger_level': danger_level,
            'conversation_id': conversation_id
        })
    except Exception as e:
        print(f"Chat Error: {e}")
        # 发生错误时返回友好的安抚消息
        return jsonify({
            'reply': '我在这里陪伴你，如果有任何不适，请寻求专业帮助。',
            'emotion_label': '平静',
            'danger_level': '低'
        }), 200

@app.route('/api/assistant/chat', methods=['POST'])
def assistant_chat():
    """悬浮智能体聊天接口：后端代理 Dify，避免前端暴露密钥"""
    try:
        data = request.get_json() or {}
        message = (data.get('message') or '').strip()
        conversation_id = data.get('conversation_id')

        if not message:
            return jsonify({'success': False, 'error': 'message 不能为空'}), 400
        if len(message) > 1000:
            return jsonify({'success': False, 'error': 'message 过长（最多1000字）'}), 400

        if not Config.DIFY_API_KEY:
            return jsonify({'success': False, 'error': 'Dify API Key 未配置（请检查 .env）'}), 500

        payload = {
            'inputs': {},
            'query': message,
            'response_mode': 'blocking',
            'user': str(session.get('user_id') or session.get('username') or 'guest')
        }
        if conversation_id:
            payload['conversation_id'] = conversation_id

        headers = {
            'Authorization': f'Bearer {Config.DIFY_API_KEY}',
            'Content-Type': 'application/json'
        }

        resp = requests.post(
            Config.DIFY_API_URL,
            headers=headers,
            json=payload,
            timeout=Config.DIFY_TIMEOUT
        )

        if resp.status_code >= 400:
            err_msg = _extract_dify_error_message(resp)

            if conversation_id and _is_conversation_not_exists(err_msg):
                payload.pop('conversation_id', None)
                resp_retry = requests.post(
                    Config.DIFY_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=Config.DIFY_TIMEOUT
                )
                if resp_retry.status_code < 400:
                    result = resp_retry.json()
                    return jsonify({
                        'success': True,
                        'reply': result.get('answer', ''),
                        'conversation_id': result.get('conversation_id'),
                        'message_id': result.get('message_id')
                    })

                err_msg = _extract_dify_error_message(resp_retry)
                logger.error(f"Dify重试失败: status={resp_retry.status_code}, body={resp_retry.text}")
                return jsonify({'success': False, 'error': err_msg, 'error_type': 'server_error'}), 502

            logger.error(f"Dify请求失败: status={resp.status_code}, body={resp.text}")
            return jsonify({'success': False, 'error': err_msg, 'error_type': 'server_error'}), 502

        result = resp.json()
        return jsonify({
            'success': True,
            'reply': result.get('answer', ''),
            'conversation_id': result.get('conversation_id') or conversation_id,
            'message_id': result.get('message_id')
        })
    except requests.Timeout:
        return jsonify({'success': False, 'error': '请求超时，请稍后再试', 'error_type': 'timeout'}), 504
    except requests.ConnectionError:
        return jsonify({'success': False, 'error': '网络连接异常，请稍后重试', 'error_type': 'network_error'}), 502
    except Exception as e:
        log_error(e, context='assistant_chat')
        return jsonify({'success': False, 'error': '系统繁忙，请稍后重试', 'error_type': 'server_error'}), 500

@app.route('/api/assistant/health')
def assistant_health():
    """智能体配置健康检查"""
    return jsonify({
        'success': True,
        'dify_url': Config.DIFY_API_URL,
        'key_configured': bool(Config.DIFY_API_KEY),
        'timeout': Config.DIFY_TIMEOUT
    })

@app.route('/api/assistant/chat/stream', methods=['POST'])
def assistant_chat_stream():
    """悬浮智能体流式聊天接口（SSE）"""
    data = request.get_json() or {}
    message = (data.get('message') or '').strip()
    conversation_id = data.get('conversation_id')

    if not message:
        return jsonify({'success': False, 'error': 'message 不能为空'}), 400
    if len(message) > 1000:
        return jsonify({'success': False, 'error': 'message 过长（最多1000字）'}), 400

    if not Config.DIFY_API_KEY:
        return jsonify({'success': False, 'error': 'Dify API Key 未配置（请检查 .env）'}), 500

    payload = {
        'inputs': {},
        'query': message,
        'response_mode': 'streaming',
        'user': str(session.get('user_id') or session.get('username') or 'guest')
    }
    if conversation_id:
        payload['conversation_id'] = conversation_id

    headers = {
        'Authorization': f'Bearer {Config.DIFY_API_KEY}',
        'Content-Type': 'application/json'
    }

    def event_stream():
        current_conversation = conversation_id
        done_sent = False

        def run_stream_request(req_payload):
            return requests.post(
                Config.DIFY_API_URL,
                headers=headers,
                json=req_payload,
                timeout=Config.DIFY_TIMEOUT,
                stream=True
            )

        def emit_done_if_needed():
            nonlocal done_sent
            if not done_sent:
                payload_done = {'conversation_id': current_conversation}
                done_sent = True
                return f"event: done\ndata: {json.dumps(payload_done, ensure_ascii=False)}\n\n"
            return None

        for attempt in range(2):
            try:
                request_payload = dict(payload)
                with run_stream_request(request_payload) as resp:
                    if resp.status_code >= 400:
                        err_msg = _extract_dify_error_message(resp)

                        if request_payload.get('conversation_id') and _is_conversation_not_exists(err_msg):
                            request_payload.pop('conversation_id', None)
                            with run_stream_request(request_payload) as retry_resp:
                                if retry_resp.status_code >= 400:
                                    retry_msg = _extract_dify_error_message(retry_resp)
                                    yield f"event: error\ndata: {json.dumps({'error': retry_msg, 'error_type': 'server_error'}, ensure_ascii=False)}\n\n"
                                    break

                                for line in retry_resp.iter_lines(decode_unicode=True):
                                    if not line or not line.startswith('data:'):
                                        continue
                                    raw = line[5:].strip()
                                    if not raw:
                                        continue
                                    if raw == '[DONE]':
                                        done_evt = emit_done_if_needed()
                                        if done_evt:
                                            yield done_evt
                                        return

                                    try:
                                        evt = json.loads(raw)
                                    except json.JSONDecodeError:
                                        continue

                                    event_name = evt.get('event', '')
                                    if evt.get('conversation_id'):
                                        current_conversation = evt.get('conversation_id')

                                    if event_name in ('message', 'agent_message'):
                                        token_text = evt.get('answer', '')
                                        if token_text:
                                            yield f"event: token\ndata: {json.dumps({'text': token_text}, ensure_ascii=False)}\n\n"
                                    elif event_name in ('message_end', 'agent_message_end'):
                                        done_evt = emit_done_if_needed()
                                        if done_evt:
                                            yield done_evt
                                        return

                                done_evt = emit_done_if_needed()
                                if done_evt:
                                    yield done_evt
                                return

                        yield f"event: error\ndata: {json.dumps({'error': err_msg, 'error_type': 'server_error'}, ensure_ascii=False)}\n\n"
                        break

                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith('data:'):
                            continue

                        raw = line[5:].strip()
                        if not raw:
                            continue

                        if raw == '[DONE]':
                            done_evt = emit_done_if_needed()
                            if done_evt:
                                yield done_evt
                            return

                        try:
                            evt = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        event_name = evt.get('event', '')
                        if evt.get('conversation_id'):
                            current_conversation = evt.get('conversation_id')

                        if event_name in ('message', 'agent_message'):
                            token_text = evt.get('answer', '')
                            if token_text:
                                yield f"event: token\ndata: {json.dumps({'text': token_text}, ensure_ascii=False)}\n\n"
                        elif event_name in ('message_end', 'agent_message_end'):
                            done_evt = emit_done_if_needed()
                            if done_evt:
                                yield done_evt
                            return

            except requests.Timeout:
                if attempt == 1:
                    yield f"event: error\ndata: {json.dumps({'error': '请求超时，请稍后再试', 'error_type': 'timeout'}, ensure_ascii=False)}\n\n"
            except requests.ConnectionError as e:
                if attempt == 1:
                    log_error(e, context='assistant_chat_stream_connection')
                    yield f"event: error\ndata: {json.dumps({'error': '连接被中断，请稍后再试', 'error_type': 'network_error'}, ensure_ascii=False)}\n\n"
            except Exception as e:
                log_error(e, context='assistant_chat_stream')
                yield f"event: error\ndata: {json.dumps({'error': '系统繁忙，请稍后重试', 'error_type': 'server_error'}, ensure_ascii=False)}\n\n"
                break

        done_evt = emit_done_if_needed()
        if done_evt:
            yield done_evt

    return Response(event_stream(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })

@app.route('/hybridaction/zybTrackerStatisticsAction')
def zyb_tracker_statistics_action():
    """兼容第三方埋点 JSONP 请求，避免控制台 404 报错"""
    callback = request.args.get('_callback') or request.args.get('callback') or ''
    payload = {'ret': 0, 'msg': 'ok'}
    if callback:
        safe_callback = ''.join(ch for ch in callback if ch.isalnum() or ch in '._$')
        body = f"{safe_callback}({json.dumps(payload, ensure_ascii=False)})"
        return Response(body, mimetype='application/javascript')
    return jsonify(payload)

# --- 对话历史 API ---
@app.route('/api/dialogue/history')
def get_dialogue_history():
    """获取对话历史"""
    try:
        conversation_id = request.args.get('conversation_id', 'default')
        
        conn = get_db_connection()
        
        # 尝试按会话ID获取，如果用户未登录则返回空
        if 'user_id' in session:
            history = conn.execute('''
                SELECT * FROM dialogue_history
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY created_at ASC
            ''', (session['user_id'], conversation_id)).fetchall()
            conn.close()
            return jsonify([dict(h) for h in history])
        else:
            conn.close()
            return jsonify([])
    except Exception as e:
        log_error(e, context="get_dialogue_history")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dialogue/conversations')
def get_conversations():
    """获取用户的所有对话会话列表"""
    try:
        if 'user_id' not in session:
            return jsonify([])
        
        conn = get_db_connection()
        # 获取用户的所有会话，按最新消息时间排序
        conversations = conn.execute('''
            SELECT d.conversation_id,
                   MAX(d.created_at) as last_message_time,
                   COUNT(*) as message_count,
                   (
                       SELECT message
                       FROM dialogue_history first_msg
                       WHERE first_msg.user_id = d.user_id
                         AND first_msg.conversation_id = d.conversation_id
                         AND first_msg.role = 'user'
                       ORDER BY first_msg.id ASC
                       LIMIT 1
                   ) as first_user_message
            FROM dialogue_history d
            WHERE d.user_id = ?
            GROUP BY d.conversation_id
            ORDER BY last_message_time DESC
            LIMIT 50
        ''', (session['user_id'],)).fetchall()
        conn.close()
        result = []
        for c in conversations:
            item = dict(c)
            item['title'] = build_conversation_title(item.get('first_user_message'))
            result.append(item)
        return jsonify(result)
    except Exception as e:
        log_error(e, context="get_conversations")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dialogue/conversations', methods=['POST'])
def create_conversation():
    """创建新对话会话"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "请先登录"}), 401
        
        conversation_id = str(uuid.uuid4())[:8]
        
        return jsonify({"success": True, "conversation_id": conversation_id})
    except Exception as e:
        log_error(e, context="create_conversation")
        return jsonify({"error": str(e)}), 500

# --- 情绪统计接口 ---
@app.route('/api/emotion-stats')
@login_required
def emotion_stats():
    """统计当前用户对话历史中各情绪标签出现次数"""
    try:
        user_id = session['user_id']
        conn = get_db_connection()
        rows = conn.execute('''
            SELECT emotion_label, COUNT(*) as count
            FROM dialogue_history
            WHERE user_id = ? AND role = 'user' AND emotion_label IS NOT NULL
            GROUP BY emotion_label
            ORDER BY count DESC
        ''', (user_id,)).fetchall()
        conn.close()
        stats = [{'emotion': row['emotion_label'], 'count': row['count']} for row in rows]
        return jsonify({'stats': stats})
    except Exception as e:
        log_error(e, context='emotion_stats')
        return jsonify({'stats': []}), 500

@app.route('/api/dialogue/clear', methods=['POST'])
def clear_dialogue():
    """清空当前会话的对话历史"""
    try:
        data = request.get_json() or {}
        conversation_id = data.get('conversation_id', 'default')
        
        if 'user_id' in session:
            conn = get_db_connection()
            conn.execute('''
                DELETE FROM dialogue_history
                WHERE user_id = ? AND conversation_id = ?
            ''', (session['user_id'], conversation_id))
            conn.commit()
            conn.close()
        
        # 同时清空session中的chat_history
        session.pop('chat_history', None)
        
        return jsonify({"success": True})
    except Exception as e:
        log_error(e, context="clear_dialogue")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # 启用多线程模式以避免请求阻塞
    # 添加超时设置：请求超时为60秒
    import socket
    socket.setdefaulttimeout(60)

    # 保持 use_reloader=False 避免 CUDA 模型重复加载
    app.run(port=5000, debug=True, use_reloader=False, threaded=True)
