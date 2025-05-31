import os, datetime, requests, pymysql
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from fal_client import subscribe

load_dotenv()

app = Flask(__name__)
CORS(app)

# MySQL Connection
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='vantruong',
    database='texttovideo',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# Tạo bảng history nếu chưa có
try:
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255),
                input_text TEXT,
                conversion_type VARCHAR(255),
                result VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
except Exception as e:
    print("Database init error:", e)

connection.commit()

# Lưu vào DB
def InsertFileToDatabase(username, prompt, type, result_path):
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO history (username, input_text, conversion_type, result, style, duration) VALUES (%s, %s, %s, %s)",
                (username, prompt, type, result_path))
            connection.commit()
            return True
    except Exception as e:
        print("Insert error:", e)
        return False

# API: Tạo video
@app.route('/api/texttovideo', methods=['POST'])
def TextToVideo():
    data = request.get_json()
    username = data.get('username')
    original_prompt = data.get('prompt')
    style = data.get('style')
    duration = data.get('duration')

    if not username or not original_prompt:
        return jsonify({"success": False, "error": "Thiếu username hoặc prompt"}), 400

    try:
        full_prompt = f"{style} style: {original_prompt}"
        result = subscribe("fal-ai/wan-t2v", {
            "prompt": full_prompt,
            "logs": True
        })


        video_url = result.get("video", {}).get("url")
        if not video_url:
            return jsonify({"success": False, "error": "Không có video trả về"}), 500

        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(backend_dir, "..")
        video_dir = os.path.join(project_root, "frontend", "video")
        os.makedirs(video_dir, exist_ok=True)

        filename = f"{username}_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.mp4"
        filepath = os.path.join(video_dir, filename)

        with open(filepath, "wb") as f:
            f.write(requests.get(video_url).content)

        InsertFileToDatabase(username, original_prompt, 'text_to_video', 'video/' + filename)


        return jsonify({"success": True, "file_path": 'video/' + filename})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# API: Lấy video theo username
@app.route('/api/get_ttv_data', methods=["GET"])
def get_ttv_data():
    username = request.args.get('username')
    if not username:
        return jsonify({"success": False, "error": "Thiếu username"}), 400

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM history WHERE username=%s AND conversion_type='text_to_video' ORDER BY id DESC LIMIT 50 ",
                (username,))
            result = cursor.fetchall()
            return jsonify({"success": True, "ttv_data": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# API: Xóa video
@app.route('/api/delete_video', methods=['POST'])
def delete_video():
    data = request.get_json()
    username = data.get('username')
    file_path = data.get('file_path')

    try:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM history WHERE username=%s AND result=%s", (username, file_path))
            connection.commit()

        abs_path = os.path.join(os.path.dirname(__file__), "..", "frontend", file_path)
        if os.path.exists(abs_path):
            os.remove(abs_path)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Phục vụ file video tĩnh
@app.route('/video/<path:filename>')
def serve_video(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "..", "frontend", "video"), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
