import pymysql, datetime, os
import aiohttp
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymysql.cursors import DictCursor
import logging

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


@app.route('/api/get_balance', methods=["GET"])
def get_balance():
    username = request.args.get('username')
    logger.info(f"Data: username={username}")

    if not username:
        return jsonify({"success": False, "error": "Missing username or type parameter"}), 400

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = """
                SELECT balance FROM USERS WHERE username=%s
            """
            cursor.execute(sql, (username, ))
            result = cursor.fetchone()
        
            if not result:
                return jsonify({
                    "success": False,
                    "error": "Khong co username nay" 
                }), 402
            logger.info(f"Data: {result}")
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if connection:
            connection.close()

    return jsonify({
        "success": True,
        "balance": result
    }), 200

@app.route('/api/get_transaction_history', methods=['GET'])
def get_transaction_history():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username missing"}), 400

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            result = []

            # Truy vấn từ bảng payment_transaction (nạp tiền - status = 1)
            sql_payment = """
                SELECT order_id, amount, order_desc, status, created_at
                FROM payment_transaction
                WHERE username = %s AND status = 1
                ORDER BY created_at DESC
            """
            cursor.execute(sql_payment, (username,))
            transactions = cursor.fetchall()
            for t in transactions:
                result.append({
                    "order_id": t["order_id"],
                    "amount": float(t["amount"]),
                    "order_desc": t["order_desc"],
                    "status": t["status"],
                    "created_at": str(t["created_at"])
                })

            # Truy vấn từ bảng thanhtoan (sử dụng dịch vụ)
            sql_service = """
                SELECT id, amount, type, create_at
                FROM thanhtoan
                WHERE username = %s
                ORDER BY create_at DESC
            """
            cursor.execute(sql_service, (username,))
            services = cursor.fetchall()
            for s in services:
                result.append({
                    "order_id": f"SERVICE-{s['id']}",  # Tạo order_id giả cho đúng format
                    "amount": float(s["amount"]),
                    "order_desc": f"Sử dụng dịch vụ: {s['type']}",
                    "status": 1,  # Mặc định là 1 vì đã hoàn tất
                    "created_at": str(s["create_at"])
                })

            # Sắp xếp toàn bộ giao dịch theo thời gian mới nhất
            result.sort(key=lambda x: x["created_at"], reverse=True)

            return jsonify({
                "success": True,
                "transactions": result
            }), 200

    except Exception as e:
        app.logger.error(f"Error fetching transaction history: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/recent_products', methods=['GET'])
def get_recent_products():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username missing"}), 400
    connection = None
    try:
        connection = get_db_connection()
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
        app.logger.error(f"Error fetching recent products: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_user_info', methods=['GET'])
def get_user_info():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username missing"}), 400
    connection = None
    try:
        connection = get_db_connection()
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
        app.logger.error(f"Error fetching user info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_avatar', methods=['GET'])
def get_avatar():
    username = request.args.get('username')
    connection = None
    try:
        connection = get_db_connection()
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
        app.logger.error(f"Error: {e}")
        return jsonify({
            "success" : False,
            "error" : str(e)
        }), 405
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555, debug=True)