from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import sys
import os

bp_file_reader = Blueprint('file_reader', __name__)

# Calculate the absolute path to the backend directory
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
print("Backend directory:", backend_dir)

# Add the backend directory to the sys.path
sys.path.insert(0, backend_dir)

from backend.classification import read_eeg_file, read_edf_eeg, read_mat_eeg
from backend.features import files

@bp_file_reader.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        files = request.files.getlist('file_path')
        
        if not files:
            return jsonify({"message": "No files uploaded"}), 400
        
        for file in files:
            if file.filename == '':
                return jsonify({"message": "No selected file"}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
                file_path = os.path.join(downloads_folder, filename)
                file.save(file_path)
                
                extension = file.filename.rsplit('.', 1)[1].lower()
                if file_path.lower().endswith(('.csv', '.xls', '.xlsx', '.xlsm', '.xlsb')):
                    raw, sfreq = read_eeg_file(file_path)
                elif extension == 'edf':
                    raw, sfreq = read_edf_eeg(file_path)
                elif extension == 'mat':
                    raw, sfreq, labels = read_mat_eeg(file_path)
                
                if raw is None:
                    return jsonify({"message": f"Failed to read file {filename}"}), 500

        return jsonify({"message": "All files successfully read"}), 200
    
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400
        
        # Check if the file has an allowed filename
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'edf', 'mat'}:
            # Here you would typically save the file and then process it
            filename = secure_filename(file.filename)             
            downloads_folder = os.path.expanduser("~/Downloads")  # Get the path to the "Downloads" directory
            file_path = os.path.join(downloads_folder, filename) 
            file.save(file_path)
            
            # Call the appropriate function based on file extension
            extension = file.filename.rsplit('.', 1)[1].lower()
            if extension in {'.csv', '.xls', '.xlsx', '.xlsm', '.xlsb'}:
                raw, sfreq = read_eeg_file(file_path)
            elif extension == 'edf':
                raw, sfreq = read_edf_eeg(file_path)
                files(file_path)
            elif extension == 'mat':
                raw, sfreq, labels = read_mat_eeg(file_path)
                files(file_path)

            if raw is not None:
                # Process was successful
                return jsonify({"message": "File successfully read"}), 200
            else:
                # Process failed
                return jsonify({"message": "Failed to read the file"}), 500
        else:
            return jsonify({"message": "Unsupported file type"}), 400
    else:
        return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'edf', 'mat', 'xlsx', 'xlsm', 'xlsb'}

