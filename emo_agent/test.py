import requests
import json

url = "https://api.dify.ai/v1/chat-messages"

headers = {
    "Authorization": "Bearer app-xmxdoL8g3aFcrf0NDi3Kqfgp",
    "Content-Type": "application/json"
}

data = {
    "inputs": {},
    "query": "我最近压力很大",
    "response_mode": "streaming",
    "user": "test_user"
}

response = requests.post(url, headers=headers, json=data, stream=True)

full_answer = ""

for line in response.iter_lines():
    if line:
        line = line.decode("utf-8")

        # 只处理 data: 开头的行
        if line.startswith("data: "):
            json_str = line[6:]

            try:
                obj = json.loads(json_str)

                # 只拿模型回复
                if obj.get("event") == "agent_message":
                    text = obj.get("answer", "")
                    print(text, end="", flush=True)
                    full_answer += text

            except:
                pass

print("\n\n完整回答：", full_answer)