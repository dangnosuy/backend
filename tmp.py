from flask import Flask, request, jsonify
from flask_cors import CORS
import bcrypt, pymysql, os
from huggingface_hub import InferenceClient


login = Flask(__name__)
CORS(login)

connection = pymysql.connect(
    host='localhost',
    user='root',
    password='vantruong',
    database='texttoeverything',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        create_user_table = """CREATE TABLE IF NOT EXISTS USERS (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            avatar_path VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )"""
        cursor.execute(create_user_table)
except Exception as e:
    print("Error creating user table: ", e)

connection.commit()

@login.route("/api/sign_up", methods=["POST"])
def sign_up():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid data."}), 400
    login.logger.info(f"Data: {data}")
    username = data.get("username")
    email = data.get("email")
    password_hash = data.get("password_hash")

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM USERS WHERE username = %s", (username,))
            check_username = cursor.fetchone() 
            cursor.execute("SELECT * FROM USERS WHERE email = %s", (email,))
            check_email = cursor.fetchone() 

            if check_username or check_email:
                return jsonify({"success": False }), 400    

            cursor.execute("INSERT INTO USERS (username, email, password_hash) VALUES (%s, %s, %s)", (username, email, password_hash))
            connection.commit()
            return jsonify({"success": True}), 200

    except Exception as e:
        login.logger.error(f"Error during registration: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500

@login.route("/api/sign_in", methods=["POST"])
def sign_in():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid data."}), 400

    username = data.get("username")
    password_hash = data.get("password_hash")

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM USERS WHERE username = %s", (username,))
            user = cursor.fetchone()

            if user and password_hash == user['password_hash']:
                return jsonify({"success": "Login successful"}), 200
            else:
                return jsonify({"error": "Incorrect username or password"}), 400

    except Exception as e:
        login.logger.error(f"Error during login: {e}")
        return jsonify({"error": f"Database error: {e}"}), 400
@login.route('/api/generate_avatar', methods=['POST']) 
def generate_avatar():
    data = request.get_json()
    username = data.get('username')
    prompt = data.get('prompt')
    login.logger.info(f"Data: {data}")
    client = InferenceClient(
        provider="hf-inference",
        api_key="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",
    )
    
    image = client.text_to_image(
        prompt,
        model="SvenN/sdxl-emoji"
    )
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(backend_dir, "..")
    img_dir = os.path.join(project_root, "frontend", "avatar")
    os.makedirs(img_dir, exist_ok=True)
    avatar_file_name = os.path.join(img_dir, f"{username}_avatar.png")
    image.save(avatar_file_name)
    try:
        with connection.cursor() as cursor:
            sql = "UPDATE USERS SET avatar_path=%s WHERE username=%s;"
            cursor.execute(sql, (f'avatar/{username}_avatar.png', username))
        connection.commit()
    except Exception as e:
        login.logger.info(f"Error: {e}")
        return jsonify({
            "success" : False,
            "error" : str(e)
        }), 402
    login.logger.info(f"File path avatar: avatar/{username}_avatar.png")
    
    return jsonify({
        "success" : True,
        "avatar_url" : f'avatar/{username}_avatar.png'
    }), 201
    # server sẽ generate ảnh, nếu người dùng oke thì dùng cái ảnh đó
@login.route('/api/get_avatar', methods=['GET'])
def get_avatar():
    username = request.args.get('username')
    try:
        with connection.cursor() as cursor:
            sql = "SELECT avatar_path FROM USERS WHERE username=%s"
            cursor.execute(sql, (username, ))
            result = cursor.fetchone()
            # Trích xuất giá trị của avatar_path
            if result:
                avatar_path = result.get('avatar_path')
            else:
                avatar_path = 'avatar/ho.jpg'
            return jsonify({
                "success" : True,
                "avatar_path" : avatar_path
            }), 203
    except Exception as e:
        login.logger.error(f"Error: {e}")
        return jsonify({
            "success" : False,
            "error" : str(e)
        }), 405
    
@login.route('/api/get_user_info', methods=['GET'])
def get_user_info():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username missing"}), 400
    try:
        with connection.cursor() as cursor:
            sql = "SELECT username, email, created_at FROM USERS WHERE username=%s"
            cursor.execute(sql, (username,))
            user = cursor.fetchone()
            if user:
                return jsonify({
                    "success": True,
                    "username": user['username'],
                    "email": user['email'],
                    "created_at": user['created_at'].strftime("%Y-%m-%d %H:%M:%S")
                }), 200
            else:
                return jsonify({"error": "User not found"}), 404
    except Exception as e:
        login.logger.error(f"Error fetching user info: {e}")
        return jsonify({"error": str(e)}), 500

@login.route('/api/recent_products', methods=['GET'])
def get_recent_products():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username missing"}), 400
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT id, input_text, conversion_type, result, timestamp
                FROM history
                WHERE username=%s
                ORDER BY id DESC
                LIMIT 5
            """
            cursor.execute(sql, (username,))
            rows = cursor.fetchall()
            # Định dạng timestamp cho chuẩn JSON
            for row in rows:
                if isinstance(row['timestamp'], datetime.datetime):
                    row['timestamp'] = row['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            return jsonify({
                "success": True,
                "products": rows
            }), 200
    except Exception as e:
        login.logger.error(f"Error fetching recent products: {e}")
        return jsonify({"error": str(e)}), 500

@payment.route('/api/get_transaction_history', methods=['GET'])
def get_transaction_history():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username missing"}), 400
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT order_id, amount, order_desc, status, created_at
                FROM payment_transaction
                WHERE username = %s
                ORDER BY created_at DESC
            """
            cursor.execute(sql, (username,))
            transactions = cursor.fetchall()
            if transactions:
                # Format lại dữ liệu cho đẹp
                result = []
                for t in transactions:
                    result.append({
                        "order_id": t["order_id"],
                        "amount": float(t["amount"]),
                        "order_desc": t["order_desc"],
                        "status": t["status"],
                        "created_at": str(t["created_at"])
                    })
                return jsonify({
                    "success": True,
                    "transactions": result
                }), 200
            else:
                return jsonify({
                    "success": True,
                    "transactions": []
                }), 200
    except Exception as e:
        payment.logger.error(f"Error fetching transaction history: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    login.run(host='0.0.0.0', port=5550, debug=True)