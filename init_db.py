import sqlite3

def init_multi_dim_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # 1. 重建题库表
    cursor.execute('DROP TABLE IF EXISTS questions')
    cursor.execute('''
        CREATE TABLE questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            category TEXT,
            scale_type TEXT,
            weight INTEGER DEFAULT 1
        )
    ''')

    # 2. 题库数据构建
    questions_data = [
        # --- PHQ-9 (抑郁: 保持原始问法) ---
        ("做事提不起劲头或没有兴趣？", "Depression", "PHQ-9"),
        ("感到心情低落、沮丧或绝望？", "Depression", "PHQ-9"),
        ("入睡困难、睡得不稳或过多？", "Depression", "PHQ-9"),
        ("感觉疲累或没什么精神？", "Depression", "PHQ-9"),
        ("胃口不好或吃得太多？", "Depression", "PHQ-9"),
        ("觉得自己很失败，或让自己及家人失望？", "Depression", "PHQ-9"),
        ("对事物专注有困难，例如看报纸或看电视？", "Depression", "PHQ-9"),
        ("动作或说话速度缓慢到别人已经察觉？", "Depression", "PHQ-9"),
        ("有自伤的想法，或想以某种方式让自己死掉？", "Depression", "PHQ-9"),

        # --- GAD-7 (焦虑: 保持原始问法) ---
        ("感到紧张、不安或急躁？", "Anxiety", "GAD-7"),
        ("无法停止或控制忧虑？", "Anxiety", "GAD-7"),
        ("对各种各样的事情担忧过多？", "Anxiety", "GAD-7"),
        ("很难放松下来？", "Anxiety", "GAD-7"),
        ("由于不安而无法静坐？", "Anxiety", "GAD-7"),
        ("容易变得烦躁或易怒？", "Anxiety", "GAD-7"),
        ("感到好像有什么可怕的事会发生？", "Anxiety", "GAD-7"),

        # --- Sleep (睡眠质量 - 统一问法适配 '从未' 到 '总是') ---
        ("觉得入睡困难，躺在床上超过半小时仍很清醒？", "Sleep", "PSS-STYLE"),
        ("在半夜或凌晨惊醒，且难以再次入睡？", "Sleep", "PSS-STYLE"),
        ("因为睡眠质量差而感到白天精神恍惚或疲惫？", "Sleep", "PSS-STYLE"),
        ("需要借助药物、褪黑素等方式辅助入睡？", "Sleep", "PSS-STYLE"),

        # --- Pressure (压力感知 - 统一问法) ---
        ("感到无法控制生活中重要的事情？", "Pressure", "PSS-STYLE"),
        ("感到压力大到无法应对？", "Pressure", "PSS-STYLE"),
        ("感到琐事堆积如山，超出了你的处理能力？", "Pressure", "PSS-STYLE"),
        ("很难静下心来放松，总觉得有事情悬而未决？", "Pressure", "PSS-STYLE"),

        # --- Social (社交状态 - 扩展维度) ---
        ("在社交场合（如聚会、开会）感到局促不安？", "Social", "PSS-STYLE"),
        ("担心别人对自己评价不高或产生误解？", "Social", "PSS-STYLE"),
        ("觉得与周围的人有隔阂，缺乏深层联系？", "Social", "PSS-STYLE"),

        # --- Self (自我价值 - 扩展维度) ---
        ("对自己正在做的事情失去信心，产生挫败感？", "Self", "PSS-STYLE"),
        ("觉得自己不如周围的人，产生自卑心理？", "Self", "PSS-STYLE"),
        ("觉得自己的努力没有得到应有的认可？", "Self", "PSS-STYLE")
    ]

    cursor.executemany(
        'INSERT INTO questions (text, category, scale_type) VALUES (?, ?, ?)', 
        questions_data
    )
    
    conn.commit()
    conn.close()
    print("多维度题库构建完成！总计 30 道题目。")

if __name__ == "__main__":
    init_multi_dim_db()