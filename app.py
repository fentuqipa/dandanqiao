from flask import Flask, render_template, send_from_directory, request, jsonify
from backend.src.chatbot import ChatBot

app = Flask(__name__)
chatbot = ChatBot()

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/Dandan.pdf')
def download_cv():
    return send_from_directory('.', 'Dandan.pdf')

@app.route("/answer", methods=["GET", "POST"])
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
    app.run()
