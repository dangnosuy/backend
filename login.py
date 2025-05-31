from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import pymysql, bcrypt, random, yagmail, os
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from datetime import datetime, timedelta

login = Flask(__name__)
CORS(login)

connection = pymysql.connect(
    host='localhost',
    user='dangnosuy',
    password='dangnosuy',
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
            password_hash VARCHAR(255),
            signin_google BOOLEAN NOT NULL,
            avatar_path VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            balance DECIMAL(12,2) NOT NULL DEFAULT 50000.00
        )"""
        cursor.execute(create_user_table)
except Exception as e:
    print("Error creating user table: ", e)

connection.commit()

EMAIL_SENDER = "tongxuanvuk1920@gmail.com" 
EMAIL_PASSWORD = "ftaj usgr mrgz dccw"
GOOGLE_CLIENT_ID = "2538207828-gqk1ah92p6fr28gsji3dr97hlmhjghn7.apps.googleusercontent.com" 
pending_verifications = {}  
reset_tokens = {}

def send_email(recipient_email, subject, body):
    try:
        yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_PASSWORD) 
        yag.send(recipient_email, subject, body)
        login.logger.info(f"Sent to {recipient_email}")
        return True
    except Exception as e:
        login.logger.error(f"Errors when sending mail: {e}")
        return False
    
@login.route("/api/sign_up", methods=["POST"])
def sign_up():
    data = request.get_json()
    if not data:
        return jsonify({"error": False}), 400
    login.logger.info(f"Data: {data}")
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM USERS WHERE username = %s", (username,))
            check_username = cursor.fetchone() 
            cursor.execute("SELECT * FROM USERS WHERE email = %s", (email,))
            check_email = cursor.fetchone() 

            if check_username or check_email:
                return jsonify({"error": False, "check_username": check_username, "check_email": check_email }), 400    

            verification_code = random.randint(100000, 999999)
            pending_verifications[email] = {
                "username": username,
                "password_hash": password_hash,
                "verification_code": verification_code
            }
            subject = "Email kiểm tra từ AlchemistAI"
            body = f"Mã xác nhận của bạn là: {verification_code}"
            if send_email(email, subject, body):
                return jsonify({"success": True, "message": "Verification code has been sent"}), 200
            else:
                return jsonify({"error": False, "message": "Failed to send the confirmation code"}), 400

    except Exception as e:
        login.logger.error(f"Error during registration: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500
    
@login.route("/api/verify_email", methods=["POST"])
def verify_email():
    data = request.get_json()
    if not data:
        return jsonify({"error": False}), 400
    login.logger.info(f"Data: {data}")
    email = data.get("email")
    user_code = data.get("code")

    if email in pending_verifications:
        if pending_verifications[email]["verification_code"] == int(user_code):
            try:
                with connection.cursor() as cursor:
                    username = pending_verifications[email]["username"]
                    password_hash = pending_verifications[email]["password_hash"]
                    cursor.execute("INSERT INTO USERS (username, email, password_hash, signin_google) VALUES (%s, %s, %s, %s)", (username, email, password_hash, False))
                    connection.commit()
                del pending_verifications[email]
                return jsonify({"success": True, "message": "Account created successfully"}), 200
            except Exception as e:
                login.logger.error(f"Error during register: {e}")
                return jsonify({"error": f"Database error: {e}"}), 500
        else:
            return jsonify({"error": False, "message": "Verification is incorrect"}), 400
    else:
        return jsonify({"error": False, "message": "Emails not on the list"}), 400

@login.route("/api/sign_in", methods=["POST"])
def sign_in():
    data = request.get_json()
    if not data:
        return jsonify({"error": False}), 400
    login.logger.info(f"Data: {data}")
    username = data.get("username")
    password = data.get("password")

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM USERS WHERE username = %s", (username,))
            user = cursor.fetchone()

            if user and bcrypt.checkpw(password.encode('utf-8'), user["password_hash"].encode('utf-8')):
                return jsonify({"success": True, "message": "Login Successfully"}), 200
            else:
                return jsonify({"error": False, "message": "Username or password is incorrect"}), 400

    except Exception as e:
        login.logger.error(f"Error during login: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500

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
@login.route("/api/google_login", methods=["POST"])
def google_login():
    data = request.get_json()
    if not data or 'idToken' not in data:
        return jsonify({"error": False,}), 400
    token = data['idToken']

    try:
        request_google = google_requests.Request()
        idinfo = id_token.verify_oauth2_token(token, request_google, GOOGLE_CLIENT_ID)
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')

        email = idinfo.get('email')

        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM USERS WHERE email = %s", (email,))
                user = cursor.fetchone()

                if user:
                    if user["signin_google"] == True:
                        return jsonify({"success": True, "username": user['username'], "message": "Đăng nhập bằng Google thành công"}), 200
                    else:
                        return jsonify({"error": False})
                else:
                    username = email.split('@')[0]
                    cursor.execute("INSERT INTO USERS (username, email, signin_google) VALUES (%s, %s, %s)", (username, email, True))
                    connection.commit()
                    login.logger.info("da luu")
                    return jsonify({"success": True, "username": username, "message": "Đăng nhập bằng Google thành công"}), 200

        except Exception as e:
            login.logger.error(f"Lỗi khi xử lý đăng nhập bằng Google: {e}")
            return jsonify({"error": False, "message": f"Lỗi cơ sở dữ liệu: {e}"}), 500

    except ValueError as e:
        login.logger.error(f"Mã thông báo Google không hợp lệ: {e}")
        return jsonify({"error": False, "message": "Mã thông báo Google không hợp lệ"}), 400
    
@login.route("/api/reset_password", methods=["POST"])
def reset_password():
    data = request.get_json()
    if not data:
        return jsonify({"error": False}), 400
    username = data.get('username')
    email = data.get('email')

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM USERS WHERE email = %s", (email,))
            user = cursor.fetchone()
            if username != user['username']:
                return jsonify({"error": False, "message": "Username or email is incorrect"}), 400
            if user:
                reset_token = os.urandom(16).hex()
                expiry_time = datetime.utcnow() + timedelta(hours=1)
                reset_tokens[reset_token] = {'email': email, 'expiry': expiry_time}

                reset_link =reset_link = f"http://127.0.0.1:5500/frontend/reset_password.html?token={reset_token}"
                subject = "Yêu cầu đặt lại mật khẩu cho AlchemistAI"
                body = f"Hãy nhấn vào link bên dưới để đổi mật khẩu:\n\n{reset_link}\n\nLiên kết sẽ tồn tại trong 1 giờ."

                if send_email(email, subject, body):
                    return jsonify({"success": True, "message": "Password reset link has been sent to your email address"}), 200
                else:
                    return jsonify({"error": False, "message": "Failed to send the password reset email"}), 500
            else:
                return jsonify({"error": False, "message": "Email address not found"}), 404

    except Exception as e:
        login.logger.error(f"Error during forgot password request: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500
    
@login.route("/api/reset_password_confirm", methods=["POST"])
def reset_password_confirm():
    data = request.get_json()
    if not data:
        return jsonify({"error": True, "message": "Dữ liệu không hợp lệ"}), 400
    
    token = data.get("token")
    new_password = data.get("new_password")
    login.logger.info(f"Data: {data}")
    if not token or not new_password:
        return jsonify({"error": True, "message": "Thiếu token hoặc mật khẩu"}), 400

    token_data = reset_tokens.get(token)

    if not token_data:
        return jsonify({"error": True, "message": "Token không hợp lệ hoặc đã hết hạn"}), 400
    if datetime.utcnow() > token_data["expiry"]:
        del reset_tokens[token]  
        return jsonify({"error": True, "message": "Token đã hết hạn"}), 400

    email = token_data["email"] 
    password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

    try:
        with connection.cursor() as cursor:
            cursor.execute("UPDATE USERS SET password_hash = %s WHERE email = %s", (password_hash, email))
        connection.commit()
        del reset_tokens[token] 

        return jsonify({"success": True, "message": "Đặt lại mật khẩu thành công"}), 200

    except Exception as e:
        login.logger.error(f"Lỗi cập nhật mật khẩu: {e}")
        return jsonify({"error": True, "message": f"Lỗi hệ thống: {e}"}), 500
    
if __name__ == "__main__":
    login.run(host='0.0.0.0', port=5550, debug=True)