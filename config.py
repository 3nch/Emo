"""
心愈 AI - 配置管理模块

支持从环境变量和 .env 文件加载配置
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# 尝试加载 .env 文件
def load_env():
    """加载 .env 文件中的环境变量"""
    env_file = BASE_DIR / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env()


# ============== 应用配置 ==============

class Config:
    """基础配置类"""
    
    # Flask 配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'emo-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # 数据库配置
    DATABASE_PATH = os.environ.get('DATABASE_PATH', str(BASE_DIR / 'database.db'))
    
    # 模型路径配置
    MODEL_DIR = os.environ.get('MODEL_DIR', str(BASE_DIR / 'best_depression_model'))
    EMOTION_MODEL_DIR = os.environ.get('EMOTION_MODEL_DIR', str(BASE_DIR / 'emotion_qwen25_lora_1'))
    WAV2VEC_MODEL_DIR = os.environ.get('WAV2VEC_MODEL_DIR', str(BASE_DIR / 'wav2vec'))
    
    # API 配置
    API_TIMEOUT = int(os.environ.get('API_TIMEOUT', '30'))
    DIFY_API_URL = os.environ.get('DIFY_API_URL', 'https://api.dify.ai/v1/chat-messages')
    DIFY_API_KEY = os.environ.get('DIFY_API_KEY', '')
    DIFY_TIMEOUT = int(os.environ.get('DIFY_TIMEOUT', '30'))
    FEISHU_WEBHOOK = os.environ.get('FEISHU_WEBHOOK', 'https://open.feishu.cn/open-apis/bot/v2/hook/24c7a115-660b-48b6-852c-4b1fce176231')
    VOICE_UPLOAD_MAX_MB = int(os.environ.get('VOICE_UPLOAD_MAX_MB', '20'))
    
    # 日志配置
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', str(BASE_DIR / 'app.log'))


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    DATABASE_PATH = ':memory:'


# 根据环境变量选择配置
env = os.environ.get('FLASK_ENV', 'development')
if env == 'production':
    Config = ProductionConfig
elif env == 'testing':
    Config = TestingConfig
else:
    Config = DevelopmentConfig


# ============== 便捷访问 ==============

# 导出一个配置字典，便于其他地方使用
def get_config():
    """获取当前配置"""
    return {
        'SECRET_KEY': Config.SECRET_KEY,
        'DATABASE_PATH': Config.DATABASE_PATH,
        'MODEL_DIR': Config.MODEL_DIR,
        'EMOTION_MODEL_DIR': Config.EMOTION_MODEL_DIR,
        'WAV2VEC_MODEL_DIR': Config.WAV2VEC_MODEL_DIR,
        'DEBUG': Config.DEBUG,
        'LOG_LEVEL': Config.LOG_LEVEL,
    }
