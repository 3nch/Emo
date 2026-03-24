"""
心愈 AI - 日志模块

提供统一的日志记录功能
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config import Config


def setup_logger(name='emo', log_file=None, level=None):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，None 则只输出到控制台
        level: 日志级别，默认使用配置中的级别
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 获取或创建日志记录器
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 设置日志级别
    log_level = level or getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 创建全局日志记录器
logger = setup_logger('emo', Config.LOG_FILE)


def log_user_action(action, username=None, details=None):
    """记录用户操作"""
    msg = f"USER_ACTION | action={action}"
    if username:
        msg += f" | username={username}"
    if details:
        msg += f" | details={details}"
    logger.info(msg)


def log_api_request(endpoint, method, status_code=None, duration_ms=None):
    """记录 API 请求"""
    msg = f"API_REQUEST | endpoint={endpoint} | method={method}"
    if status_code:
        msg += f" | status={status_code}"
    if duration_ms is not None:
        msg += f" | duration={duration_ms}ms"
    logger.info(msg)


def log_error(error, context=None):
    """记录错误"""
    msg = f"ERROR | {type(error).__name__}: {str(error)}"
    if context:
        msg += f" | context={context}"
    logger.error(msg)


def log_model_inference(model_name, input_size=None, duration_ms=None):
    """记录模型推理"""
    msg = f"MODEL_INFERENCE | model={model_name}"
    if input_size is not None:
        msg += f" | input_size={input_size}"
    if duration_ms is not None:
        msg += f" | duration={duration_ms}ms"
    logger.debug(msg)
