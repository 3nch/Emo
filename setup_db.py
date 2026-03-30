import sqlite3

def update_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 检查 records 表是否有 user_id
    cursor.execute('PRAGMA table_info(records)')
    columns = [info[1] for info in cursor.fetchall()]
    if 'user_id' not in columns:
        print("Adding user_id column to records table...")
        cursor.execute('ALTER TABLE records ADD COLUMN user_id INTEGER REFERENCES users(id)')
    
    conn.commit()
    conn.close()
    print("Database updated successfully")

if __name__ == '__main__':
    update_db()
