import pymysql, datetime, os
import aiohttp
import asyncio
import fal_client
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
from pymysql.cursors import DictCursor
import logging
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'dangnosuy',
    'password': 'dangnosuy',
    'database': 'texttoeverything',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True  # Thêm để đảm bảo commit ngay lập tức
}

# Create a connection pool helper
def get_db_connection():
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

# Initialize database tables
def init_db():
    connection = None
    try:
        connection = get_db_connection()
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
            create_tym = """
                CREATE TABLE IF NOT EXISTS tym (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) NOT NULL,
                    product VARCHAR(255) NOT NULL,
                    id_history INT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (id_history) REFERENCES history(id) ON DELETE CASCADE,
                    UNIQUE KEY unique_tym (username, id_history)
                )
            """
            cursor.execute(create_history)
            cursor.execute(create_tym)
            connection.commit()
            logger.info("Tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
    finally:
        if connection:
            connection.close()

# Initialize the database tables
init_db()

def InsertFileToDatabase(username, prompt, type, result_path):
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            insert = "INSERT INTO history (username, input_text, conversion_type, result) VALUES (%s, %s, %s, %s)"
            cursor.execute(insert, (username, prompt, type, result_path))
            connection.commit()
            logger.info(f"Inserted history record: username={username}, type={type}, result={result_path}")
            return True
    except Exception as e:
        logger.error(f"Error inserting to database: {str(e)}")
        return False
    finally:
        if connection:
            connection.close()

async def call_text_to_music_api(prompt, model="facebook/musicgen-small"): # KHÔNG CÒN SỬ DỤNG ĐƯỢC
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": "Bearer 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"}
        payload = {"inputs": prompt}
        async with session.post(
            f"https://router.huggingface.co/models/{model}",
            headers=headers,
            json=payload
        ) as response:
            audio_data = await response.read()
        return audio_data
    
def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

async def generate_audio_from_prompt(prompt):
    result = fal_client.subscribe(
        "fal-ai/ace-step/prompt-to-audio",
        arguments={"prompt": prompt},
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    return result["audio"]["url"] 


@app.route('/api/texttomusic', methods=['POST', 'OPTIONS'])
async def text_to_music():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    data = request.get_json()
    logger.info(f"Received data: {data}")
    username = data.get('username')
    prompt = data.get('prompt')

    if not username or not prompt:
        return jsonify({"success": False, "error": "Missing username or prompt"}), 400

    try:
        # Gọi API bất đồng bộ
        #audio_data = await call_text_to_music_api(prompt, model="facebook/musicgen-small")
        audio_url = await generate_audio_from_prompt(prompt)

        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(backend_dir, "..")
        music_dir = os.path.join(project_root, "frontend", "music")
        os.makedirs(music_dir, exist_ok=True)

        file_name = f"{username}_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.wav"
        file_path = os.path.join(music_dir, file_name)

        urllib.request.urlretrieve(audio_url, file_path)

        relative_path = f"music/{file_name}"

        # Lưu file âm thanh

        # Lưu vào database
        if not InsertFileToDatabase(username, prompt, "text_to_music", relative_path):
            raise Exception("Failed to insert into database")

        logger.info(f"File path: {relative_path}")
        return jsonify({
            "success": True,
            "file_path": relative_path
        }), 200
    except Exception as e:
        logger.error(f"Error in TextToMusic: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get_data_user', methods=["GET"])
def get_ttm_data_by_user():
    username = request.args.get('username')
    type = request.args.get('type')
    logger.info(f"Data: username={username}, type={type}")
    if not username or not type:
        return jsonify({"success": False, "error": "Missing username or type parameter"}), 400

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = "SELECT * FROM history WHERE username=%s AND conversion_type=%s ORDER BY id DESC"
            cursor.execute(sql, (username, type))
            result = cursor.fetchall()
            logger.info(f"Data fetched successfully: {result} ...")
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if connection:
            connection.close()

    return jsonify({
        "success": True,
        "ttm_data": result  # Sửa key để nhất quán
    }), 200

@app.route('/api/get_data_all', methods=["GET"])
def get_ttm_data_all():
    type = request.args.get('type')
    username = request.args.get('username')  # Thêm username để tính is_liked
    logger.info(f"Data: type={type}, username={username}")
    if not type:
        return jsonify({"success": False, "error": "Missing type parameter"}), 400

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = """
                SELECT 
                    h.input_text, 
                    h.id, 
                    h.result, 
                    h.username AS username, 
                    COUNT(t.id) AS quantity_tym,
                    (SELECT COUNT(*) FROM tym t2 
                     WHERE t2.id_history = h.id AND t2.username = %s) > 0 AS is_liked
                FROM history h
                LEFT JOIN tym t ON h.id = t.id_history
                WHERE h.conversion_type = %s
                GROUP BY h.id, h.input_text, h.result, h.username
                ORDER BY quantity_tym DESC
            """
            cursor.execute(sql, (username, type))
            result = cursor.fetchall()
            logger.info(f"Data fetched successfully (all): {result}")
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if connection:
            connection.close()

    return jsonify({
        "success": True,
        "ttm_data": result  # Sửa key để nhất quán
    }), 200

@app.route('/api/delete_data', methods=["POST"])
def delete_data():
    data = request.get_json()
    logger.info(f"Delete data request: {data}")
    username = data.get('username')
    file_path = data.get('file_path')
    type = data.get('type')

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = "DELETE FROM history WHERE username=%s AND result=%s"
            cursor.execute(sql, (username, file_path))
            connection.commit()
        # Kiểm tra và xóa file
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", file_path)
        if os.path.exists(full_path):
            os.remove(full_path)
            logger.info(f"File deleted: {full_path}")
            return jsonify({"success": True}), 202
        else:
            logger.warning(f"File not found: {full_path}")
            return jsonify({"success": False, "error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if connection:
            connection.close()

@app.route('/api/tym', methods=['POST'])
def tym():
    data = request.get_json()
    logger.info(f"Tym request received: {data}")
    username = data.get('username')
    id = data.get('id')
    product = data.get('product')

    if not username or not id or not product:
        return jsonify({"success": False, "error": "Missing username, id, or product"}), 400

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Kiểm tra xem người dùng đã tym chưa
            sql_check = "SELECT id FROM tym WHERE username=%s AND id_history=%s"
            cursor.execute(sql_check, (username, id))
            result = cursor.fetchone()

            if result:
                # Đã tym, xóa bản ghi (bỏ tym)
                sql_delete = "DELETE FROM tym WHERE username=%s AND id_history=%s"
                cursor.execute(sql_delete, (username, id))
                connection.commit()
                logger.info(f"Tym removed: username={username}, id_history={id}")
                return jsonify({"success": True, "action": "unliked"}), 200
            else:
                # Chưa tym, thêm bản ghi mới
                sql_insert = "INSERT INTO tym (username, product, id_history) VALUES (%s, %s, %s)"
                cursor.execute(sql_insert, (username, product, id))
                connection.commit()
                logger.info(f"Tym added: username={username}, id_history={id}")
                return jsonify({"success": True, "action": "liked"}), 200
    except pymysql.err.IntegrityError as e:
        logger.warning(f"Tym already exists or invalid history_id: {str(e)}")
        return jsonify({"success": False, "error": "Invalid request or already liked"}), 400
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if connection:
            connection.close()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5554, debug=True)