import pymysql, asyncio, datetime, os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
        sql = """CREATE TABLE IF NOT EXISTS FEEDBACK (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100),
            rating INT,
            feedback_text TEXT,
            times TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
        cursor.execute(sql)
except Exception as e:
    print("Error: ", e)

connection.commit()

def CheckUsernameAndEmail(username, email):
    try:
        with connection.cursor() as cursor:
            sql = "SELECT 1 FROM USERS WHERE username = %s AND email = %s"
            cursor.execute(sql, (username, email))
            check = cursor.fetchone()
            if check is not None:
                return 1
            else:
                return 0
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return 0

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    app.logger.info("Dữ liệu nhận được từ client: %s", data)
    username = data.get('username')
    email = data.get('email')
    rating = int(data.get('rating'))
    feedback = data.get('feedback')

    if CheckUsernameAndEmail(username, email) == 0:
        return jsonify({
            "success" : False,
            "error" : "Username and email is not exists"
        })
    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO FEEDBACK (username, rating, feedback_text) VALUES (%s, %s, %s)"
            cursor.execute(sql, (username, rating, feedback))
            connection.commit()
            return jsonify({
                "success" : True,
            }), 201
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({
            "success" : False,
            "error" : str(e)
        }), 403

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5553, debug=True)