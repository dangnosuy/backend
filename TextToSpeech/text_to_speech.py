from flask import Flask, request, send_file, render_template, jsonify, send_from_directory
import asyncio, pymysql
import subprocess
import edge_tts
import random
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename
import datetime
app = Flask(__name__)
CORS(app)
# Kết nối đến database
DB_CONFIG = {
    'host': 'localhost',
    'user': 'dangnosuy',
    'password': 'dangnosuy',
    'database': 'texttoeverything',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}
# Kết nối cơ sở dữ liệu
connection = pymysql.connect(
    host='localhost',
    user='dangnosuy',
    password='dangnosuy',
    database='texttoeverything',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

def get_db_connection():
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        app.logger.error(f"Database connection error: {str(e)}")
        raise
# Tạo bảng history nếu chưa tồn tại với các trường nhất quán
try:
    with connection.cursor() as cursor:
        create_history = """
            CREATE TABLE IF NOT EXISTS history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255),
                input_text TEXT,
                conversion_type VARCHAR(255),
                result VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        cursor.execute(create_history)
except Exception as e:
    print("Error creating table: ", e)
connection.commit()
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"  # hoặc chỉ front‑end origin
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

TTS_CHOICE_COST = 12000
TTS_UPLOAD_COST = 15000
@app.route("/api/tts_choice", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text")
    language = data.get("language")
    gender = data.get("gender")
    style = data.get("style")
    username = data.get("username")
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT balance FROM users WHERE username = %s FOR UPDATE",
                (username, )
            )
            row = cursor.fetchone()
            if not row:
                return jsonify({
                    'success': False,
                    'error': 'Người dùng không tồn tại'
                }), 404
        
            balance = float(row['balance'] or 0)

            if balance < TTS_CHOICE_COST:
                return jsonify({
                    'success': False,
                    'error': f'Số dư không khả dụng. Cần nạp thêm!!'
                }), 402
            # trừ tiền
            cursor.execute("UPDATE users SET balance = balance - %s WHERE username = %s",
                (TTS_CHOICE_COST, username)
            )
            cursor.execute(
                "INSERT INTO THANHTOAN (username, type, amount) VALUES (%s, %s, %s)",
                (username, 'text_to_speech_choice', TTS_CHOICE_COST)
            )
        conn.commit()
    except Exception as e:
        app.logger.info(f"Data: {e}")
        return jsonify({
            "success" : False,
            "error" : str(e)
        })
    app.logger.info(f"Data: {data}")

    if gender == "male":
        if style == "calm":
            reference = "model/samples/nam-calm.wav"
        elif style == "cham":
            reference = "model/samples/nam-cham.wav"
        elif style == "nhanh":
            reference = "model/samples/nam-nhanh.wav"
        else:
            reference = "model/samples/nam-truyen-cam.wav"
    else:
        if style == "calm":
            reference = "model/samples/nu-calm.wav"
        elif style == "cham":
            reference = "model/samples/nu-cham.wav"
        elif style == "luuloat":
            reference = "model/samples/nu-luu-loat.wav"
        elif style == "nhannha":
            reference = "model/samples/nu-nhan-nha.wav"
        else:
            reference = "model/samples/nu-nhe-nhang.wav"
    python_path = r"C:\Users\ADMIN\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
    tts_script = r"D:\Documents\TextToEverything\backend\TextToSpeech\tts.py"
    command = [
        python_path, tts_script,
        "-language", language,
        "-input", text,
        "-reference", reference
    ]
    app.logger.info(f"Running command: {command}")
    try:
        # Chạy lệnh và lấy output
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Đảm bảo trả về chuỗi
            encoding="utf-8",  # Ép buộc mã hóa UTF-8
            errors="replace",  # Thay thế ký tự lỗi bằng ký tự mặc định
            env=env,
            cwd=r"D:\Documents\TextToEverything\backend\TextToSpeech"
        )
        stdout, stderr = process.communicate()  # Chờ lệnh hoàn thành và lấy output
        
        app.logger.info(f"Command stdout: {stdout}")
        if process.returncode != 0:
            app.logger.error(f"Command stderr: {stderr}")
            return jsonify({"success": False, "error": f"Command failed: {stderr}"}), 401
        
        output_lines = stdout.strip().split('\n')
        file_path_line = [line for line in output_lines if "Saved final file to" in line]
        if not file_path_line:
            return jsonify({"error": "Không tìm thấy file âm thanh trong output"}), 403
        
        file_path = file_path_line[0].replace("Saved final file to ", "").strip()
        app.logger.info(f"File_path: {file_path}")
        # Lưu vào database
        with connection.cursor() as cursor:
            insert = "INSERT INTO history (username, input_text, conversion_type, result) VALUES (%s, %s, %s, %s)"
            cursor.execute(insert, (username, text, "text_to_speech", file_path))
            connection.commit()
        # Trả về kết quả cho người dùng
        new_entry = {
                "username": username,
                "input_text": text,
                "conversion_type": "text_to_speech",
                "result": file_path,
                "timestamp": int(datetime.datetime.now().timestamp() * 1000)
            }
        return jsonify({"success" : True, "result": file_path, "history_item": new_entry}), 203

    except subprocess.CalledProcessError as e:
        # Nếu tts.py gặp lỗi, trả về thông báo lỗi
        app.logger.error(f"Error output: {e.stderr}")  # In stderr khi lỗi
        app.logger.error(f"Stdout (if any): {e.stdout}")  # In stdout nếu có
        error_message = e.stderr if e.stderr else "Lỗi không xác định khi chạy tts.py"
        app.logger.error(f"Error: {e}")
        return jsonify({"error": error_message}), 401
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 402
    

@app.route("/api/tts_upload", methods=["POST"])
def tts_upload():
    username = request.form.get("username")
    prompt   = request.form.get("text")  # chính là prompt người dùng nhập
    # Kiểm tra file
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT balance FROM users WHERE username = %s FOR UPDATE",
                (username, )
            )
            row = cursor.fetchone()
            if not row:
                return jsonify({
                    'success': False,
                    'error': 'Người dùng không tồn tại'
                }), 404
        
            balance = float(row['balance'] or 0)

            if balance < TTS_UPLOAD_COST:
                return jsonify({
                    'success': False,
                    'error': f'Số dư không khả dụng. Cần nạp thêm!!'
                }), 402
            # trừ tiền
            cursor.execute("UPDATE users SET balance = balance - %s WHERE username = %s",
                (TTS_UPLOAD_COST, username)
            )
            cursor.execute(
                "INSERT INTO THANHTOAN (username, type, amount) VALUES (%s, %s, %s)",
                (username, 'text_to_speech_upload', TTS_UPLOAD_COST)
            )
        conn.commit()
    except Exception as e:
        app.logger.info(f"Data: {e}")
        return jsonify({
            "success" : False,
            "error" : str(e)
        })
    UPLOAD_FOLDER = "model_tmp"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Lưu file tạm
    filename = secure_filename(f"{username}_{file.filename}")
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(save_path)
        app.logger.info(f"Saved uploaded file to: {save_path}")
    except Exception as e:
        app.logger.error(f"Cannot save file: {e}")
        return jsonify({"success": False, "error": f"Cannot save file: {e}"}), 500

    # Chuẩn bị gọi script để train/clone giọng
    python_path = r"C:\Users\ADMIN\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
    tts_script = r"D:\Documents\TextToEverything\backend\TextToSpeech\tts.py"
    command = [
        python_path, tts_script,
        "-language", "Tiếng Việt",
        "-input", prompt,
        "-reference", f"model_tmp/{filename}"
    ]
    app.logger.info(f"Running training command: {command}")

    try:
        # Chạy lệnh
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.path.dirname(tts_script),
            env=env
        )
        stdout, stderr = proc.communicate()
        app.logger.info(f"Train stdout: {stdout}")
        if proc.returncode != 0:
            app.logger.error(f"Train stderr: {stderr}")
            return jsonify({"success": False, "error": stderr.strip()}), 500

        # Giả sử script có in ra dòng: "Trained model saved to: <path>"
        out_lines = stdout.splitlines()
        saved_line = next((l for l in out_lines if "Saved final file to" in l), None)
        if not saved_line:
            return jsonify({"success": False, "error": "Không tìm thấy output của model"}), 500

        model_path = saved_line.replace("Saved final file to ", "").strip()
        app.logger.info(f"Model path: {model_path}")

        # Lưu vào history table
        with connection.cursor() as cursor:
            insert = """
                INSERT INTO history
                  (username, input_text, conversion_type, result)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert,
                           (username, prompt, "text_to_speech", model_path))
            connection.commit()

        # Trả về thông tin
        history_item = {
            "username": username,
            "input_text": prompt,
            "conversion_type": "file_to_voice_clone",
            "result": model_path,
            "timestamp": int(datetime.datetime.now().timestamp() * 1000)
        }
        return jsonify({"success": True,
                        "model_path": model_path,
                        "history_item": history_item}), 200

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500



@app.route('/mp3/<path:filename>')
def serve_mp3(filename):
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(backend_dir, "..")
    speech_dir = os.path.join(project_root, "frontend", "mp3")
    return send_from_directory(speech_dir, filename)
# API lấy lịch sử chuyển đổi
@app.route('/api/history_tts', methods=['GET'])
def get_history():
    username = request.args.get('username')
    history = [] 
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM history WHERE username=%s AND conversion_type = 'text_to_speech' ORDER BY timestamp DESC", (username,))
            history = cursor.fetchall()
    except Exception as e:
        app.logger.error(f"Error: {e}")
    # Chuyển timestamp sang mili giây
    for item in history:
        if isinstance(item['timestamp'], datetime.datetime):
            item['timestamp'] = int(item['timestamp'].timestamp() * 1000)
    return jsonify(history)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5552, debug=True)
