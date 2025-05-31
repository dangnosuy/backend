import requests
import json
import os
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # Xuất log ra console
    ]
)

app = Flask(__name__)
CORS(app)  # Cho phép CORS để giao diện gọi API
logger = app.logger
# API Key OpenRouter (lưu trong .env để bảo mật, ngày 16/04/2025)
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Đường dẫn file hướng dẫn
INSTRUCTIONS_FILE = os.path.join(os.path.dirname(__file__), "instruction.json")

# Biến đếm yêu cầu và thời gian nạp lại
request_count = 0
last_reload_time = time.time()
RELOAD_INTERVAL = 600  # 10 phút (600 giây)
RELOAD_REQUESTS = 1    # Nạp lại sau 5 yêu cầu

# Hàm nạp hướng dẫn từ file JSON
def load_instructions():
    try:
        with open(INSTRUCTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        instructions = "\n".join([f"{item['feature']}: {item['guide']}" for item in data["instructions"]])
        logger.info("Loaded instructions: %s", instructions)
        return f"You are a helpful chatbot for the 'ALChemistAI' website. Answer user questions about how to use the website based on these instructions:\n\n{instructions}\n\nAnswer in Markdown format with clear structure, using bold text (**text**), bullet points (*), and line breaks (\n\n). Provide examples, suggest prompts if relevant, and keep responses concise and professional. Redirect off-topic questions to website usage."
    except Exception as e:
        return f"Error loading instructions: {str(e)}"

# Hàm gửi yêu cầu đến OpenRouter
def send_message(messages):
    for attempt in range(3):  # Cơ chế thử lại, như lỗi Hugging Face ngày 18/04/2025
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "deepseek/deepseek-r1:free", # deepseek/deepseek-r1:free , google/gemini-2.0-flash-thinking-exp:free, deepseek/deepseek-v3-base:free
                    "messages": messages,
                })
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"Attempt {attempt+1} failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Attempt {attempt+1} error: {e}")
        time.sleep(5)
    return "Không thể kết nối sau nhiều lần thử."

chat_history = [{"role": "system", "content": load_instructions()}]
# Endpoint nhận tin nhắn người dùng
@app.route('/api/chat', methods=['POST'])
def chat():
    # Nạp lại hướng dẫn cho mỗi yêu cầu
    
    logger.info(send_message(chat_history))
    # Lấy tin nhắn người dùng
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' in request"}), 400
    
    user_message = data['message']
    chat_history.append({"role": "user", "content": user_message})
    
    # Gửi yêu cầu và lấy phản hồi
    bot_response = send_message(chat_history)
    chat_history.append({"role": "assistant", "content": bot_response})
    
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)