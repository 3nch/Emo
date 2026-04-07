import sqlite3

def fix_database():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # 创建结果记录表（适配 33 题多维度版本）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phq_score REAL,
        anxiety_score REAL,
        sleep_score REAL,
        pressure_score REAL,
        social_score REAL,      -- 新增维度
        self_score REAL,        -- 新增维度
        risk_level TEXT,
        voice_label INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ 记录表 records 已成功创建！现在可以保存测评结果了。")

if __name__ == "__main__":
    fix_database()