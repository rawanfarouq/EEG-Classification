from flask import Blueprint, render_template, request, jsonify, flash,redirect,url_for
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
            return render_template('alert.html', message="No files uploaded", alert_type='warning')


        # Initialize a flag to check if all files are processed successfully
        all_files_processed_successfully = True

        for file in files:
            if file.filename == '':
                all_files_processed_successfully = False
                return render_template('alert.html', message="No selected files", alert_type='warning')

            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                downloads_folder = os.path.expanduser("~/Downloads")
                file_path = os.path.join(downloads_folder, filename)
                file.save(file_path)

                # Process the file based on its extension
                extension = file.filename.rsplit('.', 1)[1].lower()
                try:
                    if extension in {'csv', 'xls', 'xlsx', 'xlsm', 'xlsb'}:
                        raw, sfreq = read_eeg_file(file_path)
                    elif extension == 'edf':
                        raw, sfreq = read_edf_eeg(file_path)
                    elif extension == 'mat':
                        raw, sfreq, labels = read_mat_eeg(file_path)
                    else:
                        raise ValueError("Unsupported file type")
                    
                    if raw is None:
                        raise ValueError(f"Failed to read file {filename}")
                
                except ValueError as e:
                    # If there's an error, remove the file and set the flag to False
                    os.remove(file_path)
                    all_files_processed_successfully = False
                    return render_template('alert.html', message=str(e), alert_type='danger')

            
            else:
                all_files_processed_successfully = False
                return render_template('alert.html', message="Unsupported file type", alert_type='danger')


        if all_files_processed_successfully:
            # All files were processed successfully
           return render_template('alert.html', message="All files successfully read", alert_type='success')

        else:
            # Not all files were processed, return an appropriate message
            return render_template('alert.html', message="Some files were not processed successfully", alert_type='warning')

    else:
        return render_template("upload.html")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'edf', 'mat', 'xlsx', 'xlsm', 'xlsb'}

