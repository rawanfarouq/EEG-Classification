# views.py

from flask import Blueprint, render_template, request, jsonify
from test2 import process_folder

views = Blueprint('views', __name__)

@views.route("/")
def home():
    return render_template("index.html")

@views.route("/process-folder", methods=["POST"])
def process_folder_route():
    folder_path = request.form.get('folder_path')
    try:
        message = process_folder(folder_path)
        return jsonify({'message': message, 'success': True})
    except Exception as e:
        return jsonify({'message': str(e), 'success': False})