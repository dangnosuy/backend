import pymysql, datetime, os
import aiohttp
import asyncio
import aiomysql
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import logging

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
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
                UNIQUE KEY unique_tym (username, id_history) -- Đảm bảo mỗi người dùng chỉ tym một lần
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
            return True
    except Exception as e:
        logger.error(f"Error inserting to database: {str(e)}")
        return False
    finally:
        if connection:
            connection.close()

async def call_text_to_image_api(prompt, model="black-forest-labs/FLUX.1-dev"):
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": "Bearer TOKEN"}
        payload = {"inputs": prompt}
        async with session.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                image_data = await response.read()
                return Image.open(io.BytesIO(image_data))
            else:
                error_text = await response.text()
                raise Exception(f"API error: {response.status} - {error_text}")
            
IMAGE_GENERATION_COST = 10000
@app.route('/api/texttoimage', methods=['POST', 'OPTIONS'])
async def text_to_image():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    data = request.get_json()
    app.logger.info("Dữ liệu nhận được từ client: %s", data)
    username = data.get('username')
    prompt = data.get('prompt')

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

            if balance < IMAGE_GENERATION_COST:
                return jsonify({
                    'success': False,
                    'error': f'Số dư không khả dụng. Cần nạp thêm!!'
                }), 402
            # trừ tiền
            cursor.execute("UPDATE users SET balance = balance - %s WHERE username = %s",
                (IMAGE_GENERATION_COST, username)
            )
            cursor.execute(
                "INSERT INTO THANHTOAN (username, type, amount) VALUES (%s, %s, %s)",
                (username, 'text_to_image', IMAGE_GENERATION_COST)
            )
        conn.commit()
        try:
        # Gọi API bất đồng bộ
            image = await call_text_to_image_api(prompt, model="black-forest-labs/FLUX.1-dev")

            backend_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(backend_dir, "..")
            img_dir = os.path.join(project_root, "frontend", "img")
            os.makedirs(img_dir, exist_ok=True)

        # Tạo tên file với username và timestamp
            file_name = f"{username}_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.png"
            file_path = os.path.join(img_dir, file_name)

        # Lưu ảnh
            image.save(file_path, format="PNG")
        except Exception as e:
            app.logger.error(f"Error: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    except Exception as e:
        app.logger.error(f"Lỗi không xác định trong route: {e}")
        return jsonify({"success": False, "error": "Lỗi hệ thống không xác định."}), 500
    
    InsertFileToDatabase(username, prompt, "text_to_image", "img/" + file_name)
    logger.info(f"File path: img/{file_name}")
    return jsonify({
        "success": True,
        "result_path": file_path,
    }), 200

@app.route('/api/get_data_user', methods=["GET"])
def get_tti_data_user():
    username = request.args.get('username')
    type = request.args.get('type')
    logger.info(f"Data: username={username} & type={type}")

    if not username or not type:
        return jsonify({"success": False, "error": "Missing username or type parameter"}), 400

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = """
                SELECT result
                FROM history
                WHERE username=%s AND conversion_type=%s
                ORDER BY timestamp DESC
            """
            cursor.execute(sql, (username, type))
            result = cursor.fetchall()
            list_history = [row['result'].replace('\\', '/') for row in result]
            logger.info(f"Data: {list_history}")
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if connection:
            connection.close()

    return jsonify({
        "success": True,
        "tts_data": list_history
    }), 200

@app.route('/api/get_data_all', methods=['GET'])
def get_tti_data_all():
    type = request.args.get('type')
    username = request.args.get('username')  # Thêm username từ query
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
                    h.username, 
                    COUNT(t.id) as quantity_tym,
                    EXISTS (
                        SELECT 1 FROM tym t2 
                        WHERE t2.id_history = h.id AND t2.username = %s
                    ) AS is_liked
                FROM history h
                LEFT JOIN tym t ON h.id = t.id_history
                WHERE h.conversion_type=%s
                GROUP BY h.id, h.input_text, h.result, h.username
                ORDER BY quantity_tym DESC
            """
            cursor.execute(sql, (username, type))
            result = cursor.fetchall()
            logger.info(f"Data fetched successfully")
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if connection:
            connection.close()

    return jsonify({
        "success": True,
        "tts_data": result
    }), 200

@app.route('/api/delete_data', methods=["POST"])
def delete_data():
    data = request.get_json()
    app.logger.info(data)
    username = data.get('username')
    file_path = data.get('file_path')
    type = data.get('type')

    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM history WHERE username=%s AND result=%s AND type=%s"
            cursor.execute(sql, (username, file_path, type))
            connection.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"success": True}), 202
        else:
            return jsonify({"success": False, "error": "File not found"}), 201
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 401

@app.route('/api/tym', methods=['POST'])
def tym():
    data = request.get_json()
    logger.info(f"Tym request received: {data}")
    username = data.get('username')
    id = data.get('id')  # id_history
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
    app.run(host='0.0.0.0', port=5551, debug=True)