from flask import Blueprint, render_template, request, jsonify, flash,redirect,url_for, session
from werkzeug.utils import secure_filename
import sys
import os
import mne
import numpy as np
import tempfile
import pandas as pd 
import traceback

bp_file_reader = Blueprint('file_reader', __name__)

# Calculate the absolute path to the backend directory
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
print("Backend directory:", backend_dir)

# Add the backend directory to the sys.path
sys.path.insert(0, backend_dir)

from backend.classification import read_eeg_file, read_edf_eeg, read_mat_eeg, csv_identification,processed_data_keywords,csv_modeling,preprocess_raw_eeg,extract_features_csp,mat_modeling
from backend.features import files

@bp_file_reader.route('/upload', methods=['GET', 'POST'])
def upload():
    session.pop('csv_messages', None)

    if request.method == "POST":
        files = request.files.getlist('file_path')
        mat_file_paths = []
        
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
                        #raw, sfreq, labels = read_mat_eeg(file_path)
                        mat_file_paths.append(file_path)

                    else:
                        raise ValueError("Unsupported file type")
                    
                    if extension!='mat':
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

        if mat_file_paths:
            session['mat_file_paths'] = mat_file_paths
            print("MAT file paths stored in session:", mat_file_paths)
            return render_template('success_and_redirect.html', message="All files successfully read", redirect_url=url_for('mat_classification'))

        

        if all_files_processed_successfully:

            file_paths = [os.path.join(downloads_folder, secure_filename(file.filename)) for file in files if allowed_file(file.filename)]
        # Check if any of the uploaded files are CSV
            csv_uploaded = any(file.filename.lower().endswith('csv') for file in files)
        
            if csv_uploaded:
                messages, csv_only = csv_identification(file_paths, processed_data_keywords)
                # Store messages in session
                session['csv_messages'] = messages
                # Redirect to the csv_files page
                return render_template('success_and_redirect.html', message="All files successfully read", redirect_url=url_for('csv_files'))
            else:
                # If no CSV files were uploaded, render the success alert as normal
                return render_template('success_and_redirect.html', message="All files successfully read", redirect_url=url_for('mat_classification'))
    else:
        return render_template("upload.html")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'edf', 'mat', 'xlsx', 'xlsm', 'xlsb'}

@bp_file_reader.route('/calculate-accuracies', methods=['POST'])
def calculate_accuracies():
    if 'csv_messages' in session:
        try:
            accuracies, progress_updates = csv_modeling()  # csv_modeling now returns both accuracies and progress
            return jsonify({
                'accuracies': accuracies,
                'progress_updates': progress_updates
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'No CSV files found in session.'}), 400
    
@bp_file_reader.route('/progress')
def progress():
    progress = session.get('progress', 0)
    return jsonify({'progress': progress})  


@bp_file_reader.route('/mat_classification', methods=['GET', 'POST'])      
def mat_classification():

    if request.method=='POST':

        try:

            mat_file_paths = session.get('mat_file_paths', [])
            all_preprocessing_steps = []
            

            for file_path in mat_file_paths:
                print("Processing file paths:", mat_file_paths)
                filename = os.path.basename(file_path)
                filename_parts = filename.split('_')
                subject_identifier = '_'.join(filename_parts[1:3])  # sub-XXX_ses-XX

                raw, sfreq, labels = read_mat_eeg(file_path)
                preprocessed_raw, preprocessing_steps = preprocess_raw_eeg(raw, sfreq, subject_identifier)
                for step in preprocessing_steps:
                            if step not in all_preprocessing_steps:
                                all_preprocessing_steps.append(step)    
            
            
            # Convert labels to list if it's an ndarray
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()

            # Store labels in session
            session['labels'] = labels
            some_directory = "E:\\EEG-Classification"  # Ensure this directory exists and has write permissions
            raw_file_path = os.path.join(some_directory, f"{subject_identifier}_preprocessed_raw.fif")
            preprocessed_raw.save(raw_file_path, overwrite=True)

            # Store the reference to this file in the session
            session['preprocessed_raw_file_path'] = raw_file_path
            session['subject_identifier'] = subject_identifier
            session['labels'] = labels
            # Clear the stored file paths after processing
            session.pop('mat_file_paths', None)
            # Retrieve preprocessing steps from session
            #preprocessing_steps = session.get('preprocessing_steps', [])
            # Store the collected preprocessing steps in the session
            session['preprocessing_steps'] = all_preprocessing_steps  # Store in session

            # Retrieve preprocessing steps from session, not from a local variable
            preprocessing_steps = session.get('preprocessing_steps', [])  # Retrieve from session

            # Function to extract the session number for sorting
            def get_session_number(step_description):
                # This assumes the format "Interpolated bad channels for sub-XXX_ses-XX"
                parts = step_description.split('_ses-')
                session_part = parts[1]
                session_number = int(session_part.split(':')[0])  # Convert to integer for correct numeric sorting
                return session_number

            # Reorder the steps
            ica_filter_steps = [step for step in preprocessing_steps if "High-pass filtered" in step or "Applied ICA" in step]
            
            # Sort interpolated steps by extracting the session number
            interpolated_steps = sorted(
                [step for step in preprocessing_steps if step.startswith("Interpolated bad channels for")],
                key=get_session_number
            )

            other_steps = [step for step in preprocessing_steps if step not in ica_filter_steps and step not in interpolated_steps]

            # Concatenate the lists to get the desired order
            ordered_preprocessing_steps = ica_filter_steps + other_steps + interpolated_steps

            return jsonify(preprocessing_steps=ordered_preprocessing_steps)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:         
        return render_template('mat_files_class.html')
    
@bp_file_reader.route('/extract-csp-features', methods=['POST'])
def extract_csp_features():
    try:
        # Retrieve preprocessed data and other necessary variables from session
        raw_file_path = session.get('preprocessed_raw_file_path')
        if not raw_file_path or not os.path.exists(raw_file_path):
            raise FileNotFoundError("Preprocessed raw file path not found or file does not exist.")

        # Load the preprocessed Raw object from the file
        preprocessed_raw = mne.io.read_raw_fif(raw_file_path, preload=True)
        subject_identifier = session.get('subject_identifier')
        labels = session.get('labels')

        # Extract CSP features
        features_message,features_df = extract_features_csp(preprocessed_raw, 250, labels, epoch_length=1.0)

        # Save features_df to a temporary file
        _, temp_file_path = tempfile.mkstemp(suffix='.csv')  # Creates a temp file
        features_df.to_csv(temp_file_path, index=False)  # Save DataFrame to CSV

        # Store the path of the temporary file in the session
        session['features_file_path'] = temp_file_path

        # Convert DataFrame to JSON
        features_json = features_df.to_json(orient='records')

        return jsonify(features=features_json, message=features_message)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@bp_file_reader.route('/perform_mat_modeling', methods=['POST'])
def perform_mat_modeling():
    try:
        features_file_path = session.get('features_file_path')
        if not features_file_path or not os.path.exists(features_file_path):
            raise FileNotFoundError("Features file path not found or file does not exist.")

        # Read features_df from the CSV file
        features_df = pd.read_csv(features_file_path)
        subject_identifier = session.get('subject_identifier')
        labels = session.get('labels')


        # Call the mat_modeling function
        accuracy_messages = mat_modeling(subject_identifier, features_df, labels)

        # Store the accuracy messages in the session or pass to the template
        session['accuracy_messages'] = accuracy_messages

        return jsonify(accuracy=accuracy_messages)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500











