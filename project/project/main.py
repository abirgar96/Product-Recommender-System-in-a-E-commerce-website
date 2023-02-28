# main.py
# from . import chat
import chat
from autocorrect import Speller
from flask import Flask,Blueprint, render_template, request,jsonify
from flask_login import login_required, current_user
from datetime import datetime
import os
import json

app = Flask(__name__)
main = Blueprint('main', __name__)


@app.route("/chat", methods=['GET','POST'])
def get_bot_response():
    spell = Speller()
    userText = request.args.get('msg')
    if request.method =='POST':
        content = request.json
        userText=content["usertext"]
        response= str(chat.chatbot_response(userText))
        res={"rep":str(response)}
        return jsonify(res)
    response= str(chat.chatbot_response(userText))
    if response in ["Sorry, can't understand you", "Please give me more info", "Not sure I understand"]:
        response= str(chat.chatbot_response(spell(userText)))

    return {"msg":str(response)}




if __name__ == "__main__":
    app.run(debug=True)
