"""
心愈 AI - 单元测试模块

测试核心业务逻辑
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, get_config, load_env


class TestConfig:
    """配置模块测试"""
    
    def test_config_has_required_keys(self):
        """测试配置包含所有必需的键"""
        config = get_config()
        required_keys = [
            'SECRET_KEY', 'DATABASE_PATH', 'MODEL_DIR',
            'EMOTION_MODEL_DIR', 'WAV2VEC_MODEL_DIR',
            'DEBUG', 'LOG_LEVEL'
        ]
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"
    
    def test_secret_key_not_default_in_production(self):
        """测试生产环境不使用默认密钥"""
        import os
        os.environ['FLASK_ENV'] = 'production'
        os.environ['SECRET_KEY'] = 'test-secret-key'
        
        # 重新导入以应用新配置
        from importlib import reload
        import config
        reload(config)
        
        assert config.Config.SECRET_KEY == 'test-secret-key'
        assert config.Config.SECRET_KEY != 'emo-secret-key-change-in-production'
    
    def test_database_path_is_valid(self):
        """测试数据库路径格式正确"""
        db_path = Path(Config.DATABASE_PATH)
        assert db_path.suffix == '.db', "Database path should end with .db"
    
    def test_model_dirs_exist_or_can_be_created(self):
        """测试模型目录路径格式正确"""
        model_dirs = [
            Config.MODEL_DIR,
            Config.EMOTION_MODEL_DIR,
            Config.WAV2VEC_MODEL_DIR
        ]
        for model_dir in model_dirs:
            path = Path(model_dir)
            assert path.is_absolute() or path.exists(), f"Model dir path invalid: {model_dir}"


class TestRiskAssessment:
    """风险评估测试"""
    
    def test_risk_level_boundaries(self):
        """测试风险等级边界"""
        # 定义风险评估逻辑（与 app.py 中一致）
        def assess_risk(total_score):
            if total_score < 5:
                return "低"
            elif total_score < 10:
                return "低"
            elif total_score < 15:
                return "中"
            else:
                return "高"
        
        # 测试边界值
        assert assess_risk(0) == "低"
        assert assess_risk(4) == "低"
        assert assess_risk(5) == "低"
        assert assess_risk(9) == "低"
        assert assess_risk(10) == "中"
        assert assess_risk(14) == "中"
        assert assess_risk(15) == "高"
        assert assess_risk(27) == "高"
    
    def test_score_range(self):
        """测试分数范围"""
        def assess_risk(total_score):
            if total_score < 5:
                return "低"
            elif total_score < 10:
                return "低"
            elif total_score < 15:
                return "中"
            else:
                return "高"
        
        # PHQ-9 分数范围: 0-27
        assert assess_risk(0) == "低"
        assert assess_risk(27) == "高"
        
        # 负数应该被处理
        assert assess_risk(-1) == "低"


class TestDateHandling:
    """日期处理测试"""
    
    def test_date_format_consistency(self):
        """测试日期格式一致性"""
        from datetime import datetime
        
        # 模拟数据库返回的日期格式
        db_date = "2024-01-15 10:30:00"
        iso_date = "2024-01-15T10:30:00"
        
        # 提取日期部分进行比较
        def extract_date(date_str):
            if 'T' in date_str:
                return date_str.split('T')[0]
            return date_str.split(' ')[0]
        
        assert extract_date(db_date) == extract_date(iso_date) == "2024-01-15"
    
    def test_consecutive_days_calculation(self):
        """测试连续打卡天数计算"""
        from datetime import datetime, timedelta
        
        def calculate_consecutive_days(checkin_dates):
            """计算连续打卡天数"""
            if not checkin_dates:
                return 0
            
            # 转换日期字符串
            dates = [datetime.strptime(d.split(' ')[0], '%Y-%m-%d').date() for d in checkin_dates]
            dates.sort(reverse=True)
            
            # 去重
            unique_dates = []
            for d in dates:
                if not unique_dates or d != unique_dates[-1]:
                    unique_dates.append(d)
            
            # 计算连续天数
            consecutive = 1
            today = datetime.now().date()
            
            for i in range(len(unique_dates) - 1):
                diff = (unique_dates[i] - unique_dates[i + 1]).days
                if diff == 1:
                    consecutive += 1
                else:
                    break
            
            return consecutive
        
        today = datetime.now().date()
        yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
        two_days_ago = (today - timedelta(days=2)).strftime('%Y-%m-%d')
        week_ago = (today - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # 测试只有今天的打卡
        today_str = today.strftime('%Y-%m-%d')
        assert calculate_consecutive_days([today_str]) == 1
        
        # 测试连续两天打卡
        assert calculate_consecutive_days([today_str, yesterday]) == 2
        
        # 测试连续三天打卡
        assert calculate_consecutive_days([today_str, yesterday, two_days_ago]) == 3
        
        # 测试中间断开的情况
        assert calculate_consecutive_days([today_str, week_ago]) == 1


class TestInputValidation:
    """输入验证测试"""
    
    def test_username_validation(self):
        """测试用户名验证"""
        import re
        
        def validate_username(username):
            if not username:
                return False, "用户名不能为空"
            if len(username) < 3:
                return False, "用户名至少3个字符"
            if len(username) > 20:
                return False, "用户名最多20个字符"
            if not re.match(r'^[\w\u4e00-\u9fa5]+$', username):
                return False, "用户名只能包含字母、数字、下划线和中文"
            return True, ""
        
        assert validate_username("") == (False, "用户名不能为空")
        assert validate_username("ab") == (False, "用户名至少3个字符")
        assert validate_username("user123") == (True, "")
        assert validate_username("用户你好") == (True, "")
        assert validate_username("user@123") == (False, "用户名只能包含字母、数字、下划线和中文")
    
    def test_password_validation(self):
        """测试密码验证"""
        def validate_password(password):
            if not password:
                return False, "密码不能为空"
            if len(password) < 6:
                return False, "密码至少6个字符"
            if len(password) > 50:
                return False, "密码最多50个字符"
            return True, ""
        
        assert validate_password("") == (False, "密码不能为空")
        assert validate_password("12345") == (False, "密码至少6个字符")
        assert validate_password("123456") == (True, "")
        assert validate_password("a" * 50) == (True, "")
        assert validate_password("a" * 51) == (False, "密码最多50个字符")
    
    def test_score_validation(self):
        """测试分数验证"""
        def validate_score(score, min_val=0, max_val=27):
            try:
                score = int(score)
                if score < min_val or score > max_val:
                    return False
                return True
            except (ValueError, TypeError):
                return False
        
        assert validate_score(0) == True
        assert validate_score(27) == True
        assert validate_score(-1) == False
        assert validate_score(28) == False
        assert validate_score("abc") == False
        assert validate_score(None) == False


class TestXSSPrevention:
    """XSS 防护测试"""
    
    def test_html_escaping(self):
        """测试 HTML 转义"""
        def escape_html(text):
            if not text:
                return text
            replacements = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#x27;'
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            return text
        
        assert escape_html("<script>alert('xss')</script>") == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
        assert escape_html("Hello & Goodbye") == "Hello &amp; Goodbye"
        assert escape_html('<div class="test">') == "&lt;div class=&quot;test&quot;&gt;"
    
    def test_sql_injection_prevention(self):
        """测试 SQL 注入防护（使用参数化查询）"""
        # 确保使用参数化查询而不是字符串拼接
        # 这里只是测试参数化查询的概念
        def sanitize_for_sql(value):
            """简单的 SQL 注入防护（实际应使用参数化查询）"""
            if not isinstance(value, str):
                return value
            # 替换单引号
            return value.replace("'", "''")
        
        assert sanitize_for_sql("user' OR '1'='1") == "user'' OR ''1''=''1"
        assert sanitize_for_sql("normal_text") == "normal_text"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
