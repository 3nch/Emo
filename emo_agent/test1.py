import sys
import json
import time
import requests
import pyttsx3
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QMovie

# ======================
# 🔐 配置
# ======================


# 是否关闭SSL验证（解决EOF问题）
VERIFY_SSL = False


# ======================
# 🔊 语音引擎
# ======================
engine = pyttsx3.init()
engine.setProperty('rate', 180)


def speak(text):
    engine.say(text)
    engine.runAndWait()


# ======================
# 🌐 流式请求线程（超稳定版）
# ======================
class ChatThread(QThread):
    result_signal = pyqtSignal(str)
    stream_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str)

    def __init__(self, msg, conversation_id=None):
        super().__init__()
        self.msg = msg
        self.conversation_id = conversation_id

    def run(self):
        url = "https://api.dify.ai/v1/chat-messages"

        headers = {
            "Authorization": f"Bearer app-xmxdoL8g3aFcrf0NDi3Kqfgp",
            "Content-Type": "application/json"
        }

        data = {
            "inputs": {},
            "query": self.msg,
            "response_mode": "streaming",
            "conversation_id": self.conversation_id,
            "user": "desktop_pet"
        }

        retry = 3
        for i in range(retry):
            try:
                with requests.post(
                        url,
                        headers=headers,
                        json=data,
                        stream=True,
                        timeout=60,
                        verify=VERIFY_SSL
                ) as r:

                    if r.status_code != 200:
                        self.result_signal.emit(f"❌ 错误码: {r.status_code}\n{r.text}")
                        return

                    answer = ""

                    for line in r.iter_lines():
                        if line:
                            decoded = line.decode("utf-8")

                            if decoded.startswith("data:"):
                                try:
                                    payload = json.loads(decoded[5:])

                                    if payload.get("event") == "message":
                                        chunk = payload.get("answer", "")
                                        answer += chunk
                                        self.stream_signal.emit(chunk)

                                    if payload.get("event") == "message_end":
                                        self.conversation_id = payload.get("conversation_id")
                                        self.done_signal.emit(answer)
                                        return

                                except:
                                    continue

                return

            except Exception as e:
                if i == retry - 1:
                    self.result_signal.emit(f"❌ 网络异常: {e}")
                time.sleep(1)


# ======================
# 🧸 桌宠UI
# ======================
class Assistant(QWidget):
    def __init__(self):
        super().__init__()

        self.conversation_id = None

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(300, 200, 320, 480)

        layout = QVBoxLayout()

        # 🎞️ GIF桌宠（替换为你的角色）
        self.avatar = QLabel()
        self.movie = QMovie("pet.gif")  # 👉 放一个GIF在同目录
        self.avatar.setMovie(self.movie)
        self.movie.start()

        self.avatar.setAlignment(Qt.AlignCenter)

        # 💬 聊天显示
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        # ⌨️ 输入
        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(60)

        # 🔘 按钮
        self.btn = QPushButton("发送")
        self.btn.clicked.connect(self.send_msg)

        layout.addWidget(self.avatar)
        layout.addWidget(self.chat_display)
        layout.addWidget(self.input_box)
        layout.addWidget(self.btn)

        self.setLayout(layout)

        # 👉 点击桌宠互动
        self.avatar.mousePressEvent = self.pet_click

    # 🧠 发送消息
    def send_msg(self):
        text = self.input_box.toPlainText().strip()
        if not text:
            return

        self.chat_display.append(f"🧑 你：{text}")
        self.input_box.clear()
        self.btn.setEnabled(False)

        self.thread = ChatThread(text, self.conversation_id)
        self.thread.stream_signal.connect(self.update_stream)
        self.thread.done_signal.connect(self.finish_reply)
        self.thread.result_signal.connect(self.error_reply)
        self.thread.start()

    # 🧩 流式输出
    def update_stream(self, chunk):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(chunk)
        self.chat_display.setTextCursor(cursor)

    # ✅ 完成回复
    def finish_reply(self, full_text):
        self.chat_display.append("\n")
        self.btn.setEnabled(True)

        # 🔊 语音播放
        speak(full_text)

    # ❌ 错误
    def error_reply(self, err):
        self.chat_display.append(f"\n{err}\n")
        self.btn.setEnabled(True)

    # 🧸 点击桌宠
    def pet_click(self, event):
        self.chat_display.append("🤖：戳我干嘛～陪你聊聊天呀 😄")

    # 🖱️ 拖动
    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = event.globalPos() - self.oldPos
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()


# ======================
# 🚀 启动
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Assistant()
    win.show()
    sys.exit(app.exec_())