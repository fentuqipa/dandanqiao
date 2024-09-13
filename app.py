from flask import Flask, render_template, send_from_directory, request, jsonify
from backend.src.chatbot import ChatBot
import os
from dotenv import load_dotenv
from openai import OpenAI
# from flask_sqlalchemy import SQLAlchemy

# Load environment variables from .env file if present
load_dotenv()

class Config:
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_default_secret_key')
    DEBUG = os.environ.get('FLASK_ENV') != 'production'


app = Flask(__name__)
openai_api_key = os.environ.get('OPENAI_API_KEY')
chatbot = ChatBot(api_key=openai_api_key)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/Dandan.pdf')
def download_cv():
    return send_from_directory('.', 'Dandan.pdf')

@app.route("/answer", methods=["POST"])
def chat():
    req_data = request.get_json()
    msg = req_data["msg"]
    history = req_data["history"]
    chat_history = convert_chat_history(history)
    return chatbot.generate_response(msg, chat_history)[0]

def convert_chat_history(history):
    assert len(history) % 2 == 0
    chat_history = []
    for i in range(0, len(history), 2):
        chat_history.append((history[i], history[i+1]))
    return chat_history

if __name__ == '__main__':
    # app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
    # app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_default_secret_key')
    # db = SQLAlchemy(app)
    # # app.config.from_object(Config)
    # app.run()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
