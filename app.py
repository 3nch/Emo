import os
import json
import uuid
import sqlite3
import torch
import librosa
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from functools import wraps

# 导入情绪安抚pipeline
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline'))
from pipeline.dialogue_pipeline import DialoguePipeline

app = Flask(__name__)
app.secret_key = 'emo_secret_key_123' # 在实际生产中应使用更复杂的密钥
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
MODEL_PATH = r"E:\Emo\best_depression_model"

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

# --- 2. 数据库工具函数 ---
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    try:
        # 1. 创建用户表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                created_at DATE DEFAULT (date('now'))
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                likes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 9. 创建帖子点赞表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS post_likes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(post_id, user_id)
            )
        ''')
        
        conn.commit()
    finally:
        conn.close()

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
            return redirect(url_for('index'))
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
            
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('用户名已存在')
        except Exception as e:
            print(f"Registration Error: {e}")
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

# --- 社区页面 ---
@app.route('/community')
def community():
    return render_template('community.html')

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
            content = data.get('content', '')
            mood_tag = data.get('mood_tag', '')
            
            # 生成匿名昵称
            anonymous_names = ['倾听者', '阳光小伙伴', '心灵旅者', '温暖的人', '努力的人', '勇敢的心', '微笑面对', '宁静致远', '追光者', '破晓']
            import random
            anonymous_name = random.choice(anonymous_names) + str(random.randint(1, 999))
            
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO posts (user_id, anonymous_name, content, mood_tag) VALUES (?, ?, ?, ?)',
                (user_id, anonymous_name, content, mood_tag)
            )
            conn.commit()
            conn.close()
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    else:
        # 获取帖子列表
        conn = get_db_connection()
        posts = conn.execute('''
            SELECT p.*, 
                   (SELECT COUNT(*) FROM post_likes WHERE post_id = p.id) as like_count,
                   CASE WHEN ? > 0 AND (SELECT COUNT(*) FROM post_likes WHERE post_id = p.id AND user_id = ?) > 0 THEN 1 ELSE 0 END as liked
            FROM posts p 
            ORDER BY p.created_at DESC
        ''', (session.get('user_id', 0), session.get('user_id', 0))).fetchall()
        conn.close()
        return jsonify([dict(p) for p in posts])

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
            conn.execute('INSERT INTO post_likes (post_id, user_id) VALUES (?, ?)', (post_id, user_id))
            conn.execute('UPDATE posts SET likes = likes + 1 WHERE id = ?', (post_id,))
            liked = True
        
        conn.commit()
        new_likes = conn.execute('SELECT likes FROM posts WHERE id = ?', (post_id,)).fetchone()['likes']
        conn.close()
        
        return jsonify({"success": True, "liked": liked, "likes": new_likes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 4. 核心 API 接口 ---
@app.route('/api/questions')
def get_questions():
    conn = get_db_connection()
    questions = conn.execute('SELECT * FROM questions').fetchall()
    conn.close()
    return jsonify([dict(q) for q in questions])

@app.route('/api/submit', methods=['POST'])
def submit_assessment():
    try:
        user_answers = json.loads(request.form.get('answers', '{}'))
        audio_file = request.files.get('audio')
        
        v_label = 0
        v_conf = 0.0
        
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
        
        # --- 综合风险判定逻辑 ---
        phq = scores["Depression"]
        # 如果语音模型识别为1，或者抑郁总分超过15（PHQ-9标准中重度临界点）
        if v_label == 1 or phq >= 15:
            risk = "高风险"
        elif phq >= 7:
            risk = "中等风险"
        else:
            risk = "低风险"

        # 存入数据库 (注意：确保你的 records 表有对应的字段)
        conn.execute('''
            INSERT INTO records (
                phq_score, anxiety_score, sleep_score, pressure_score, 
                social_score, self_score, risk_level, voice_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scores["Depression"], 
            scores["Anxiety"], 
            scores["Sleep"], 
            scores["Pressure"],
            scores["Social"],
            scores["Self"],
            risk, 
            v_label
        ))
        conn.commit()
        conn.close()

        # --- 雷达图归一化处理 (折算为10分制) ---
        # 计算公式：(实际得分 / 该维度最大可能得分) * 10
        radar_data = [
            round((scores["Depression"] / 27) * 10, 1), # PHQ-9 满分 27
            round((scores["Anxiety"] / 21) * 10, 1),    # GAD-7 满分 21
            round((scores["Sleep"] / 12) * 10, 1),      # 4题 满分 12
            round((scores["Pressure"] / 12) * 10, 1),   # 4题 满分 12
            round((scores["Social"] / 9) * 10, 1),      # 3题 满分 9
            round((scores["Self"] / 9) * 10, 1),        # 3题 满分 9
            round((v_label * 7 + 2), 1)                 # 语音风险映射
        ]

        return jsonify({
            "risk_level": risk,
            "radar_data": radar_data,
            "scores": scores # 将原始分数也传回前端，方便详细显示
        })
    except Exception as e:
        print(f"Submit Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- 增加获取历史记录接口 ---
@app.route('/api/history')
def get_history():
    try:
        conn = get_db_connection()
        # 按时间倒序排列，最新的在前
        records = conn.execute('SELECT * FROM records ORDER BY created_at DESC').fetchall()
        conn.close()
        return jsonify([dict(r) for r in records])
    except Exception as e:
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
                'INSERT INTO daily_checkin (user_id, mood_score, mood_label, note) VALUES (?, ?, ?, ?)',
                (user_id, mood_score, mood_label, note)
            )
            conn.commit()
            conn.close()
            return jsonify({"success": True, "message": "打卡成功"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# --- 获取用户统计数据 API ---
@app.route('/api/stats')
def get_stats():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "请先登录"}), 401
    
    try:
        conn = get_db_connection()
        
        # 1. 获取测评记录统计
        records = conn.execute(
            'SELECT * FROM records WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        ).fetchall()
        
        # 2. 获取打卡记录
        checkins = conn.execute(
            'SELECT * FROM daily_checkin WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        ).fetchall()
        
        # 3. 计算统计数据
        total_assessments = len(records)
        total_checkins = len(checkins)
        
        # 计算连续打卡天数
        consecutive_days = 0
        if checkins:
            checkin_dates = [c['created_at'] for c in checkins]
            today = datetime.now().date()
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
            date = (datetime.now() - timedelta(days=i)).date().isoformat()
            checkin = next((c for c in checkins if c['created_at'] == date), None)
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
    import csv
    from io import StringIO
    from flask import make_response
    
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
        output = StringIO()
        
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

# --- 情绪垃圾桶聊天接口（接入情绪安抚pipeline）---
# 全局pipeline实例（延迟初始化）
_dialogue_pipeline = None

def get_dialogue_pipeline():
    global _dialogue_pipeline
    if _dialogue_pipeline is None:
        _dialogue_pipeline = DialoguePipeline()
    return _dialogue_pipeline

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    情绪垃圾桶聊天接口
    接收用户文本，调用dialogue_pipeline进行情绪分析并生成安抚回复
    """
    try:
        data = request.get_json()
        user_text = data.get('text', '').strip()
        
        if not user_text:
            return jsonify({'reply': '我在这里倾听你，请告诉我你的想法。'}), 400
        
        # 调用情绪安抚pipeline
        pipeline = get_dialogue_pipeline()
        reply = pipeline.run(user_text)
        
        return jsonify({'reply': reply})
    except Exception as e:
        print(f"Chat Error: {e}")
        # 发生错误时返回友好的安抚消息
        return jsonify({'reply': '我在这里陪伴你，如果有任何不适，请寻求专业帮助。'}), 200

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        
    # 保持 use_reloader=False 避免 CUDA 模型重复加载
    app.run(port=5000, debug=True, use_reloader=False, threaded=False)
