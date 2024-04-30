from flask import Blueprint, render_template, request, jsonify, flash,redirect,url_for, session
from werkzeug.utils import secure_filename
import sys
import os
import mne
import numpy as np
import tempfile
import pandas as pd 
import traceback, shutil,tempfile, logging


bp_file_reader = Blueprint('file_reader', __name__)

# Calculate the absolute path to the backend directory
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
print("Backend directory:", backend_dir)

# Add the backend directory to the sys.path
sys.path.insert(0, backend_dir)

from backend.classification import read_eeg_file, read_edf_eeg, read_mat_eeg, csv_identification,processed_data_keywords,csv_modeling,preprocess_raw_eeg,extract_features_csp,mat_modeling,get_label_text,read_label_conditions
from backend.classification import csv_svc_model,csv_random_model,csv_logistic_model,csv_knn_model,csv_cnn_model,load_and_predict_svc,load_and_predict_random,load_and_predict_knn,load_and_predict_cnn,load_and_predict_logisitc,csv_features
from backend.classification import mat_modeling_svc,mat_modeling_random,mat_modeling_logistic,mat_modeling_knn,mat_modeling_cnn,predict_movement,predict_movement_svc,predict_movement_random,predict_movement_logistic,predict_movement_knn,predict_movement_cnn
from backend.features import extract_features


@bp_file_reader.route('/upload', methods=['GET', 'POST'])
def upload():
    print("Entered upload")
    session.pop('csv_messages', None)
    file_path_feature=[]

    if request.method == "POST":
        files = request.files.getlist('file_path_upload')
        mat_file_paths = []
        csv_or_excel_uploaded = False  # Flag for detecting CSV or Excel files

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
                print(f"File saved to {file_path}")

                # Determine the file type and process accordingly
                extension = file.filename.rsplit('.', 1)[1].lower()
                file_path_label_original=[]
                try:
                    if extension in {'csv', 'xls', 'xlsx', 'xlsm', 'xlsb'}:
                        csv_or_excel_uploaded = True  # Set the flag if a CSV or Excel file is found
                        file_path_feature.append(file_path)
                        print(f"Calling read_eeg_file for {file_path}")
                        raw, sfreq = read_eeg_file(file_path)
                        print(f"read_eeg_file returned {len(raw)} samples for {file_path}")
                        
                    elif extension =='txt':
                        file_path_label_original.append(file_path)
                        session['file_path_label_original']=file_path_label_original

                    elif extension == 'edf':
                        raw, sfreq = read_edf_eeg(file_path)
                    elif extension == 'mat':
                        mat_file_paths.append(file_path)
                    else:
                        raise ValueError("Unsupported file type")
                    
                    if extension != 'mat' and raw is None:
                        raise ValueError(f"Failed to read file {filename}")
                
                except ValueError as e:
                    os.remove(file_path)
                    all_files_processed_successfully = False
                    return render_template('alert.html', message=str(e), alert_type='danger')
            else:
                all_files_processed_successfully = False
                return render_template('alert.html', message="Unsupported file type", alert_type='danger')

        if mat_file_paths:
            extract_features(mat_file_paths)
            session['mat_file_paths'] = mat_file_paths
            print("MAT file paths stored in session:", mat_file_paths)
            # Redirect to mat_classification only if mat files are uploaded and no other types are present
            if not csv_or_excel_uploaded:
                return render_template('success_and_redirect.html', message="All files successfully read", redirect_url=url_for('mat_classification'))
        
        if all_files_processed_successfully:
            # Redirect to csv_files if any CSV or Excel files were uploaded
            if csv_or_excel_uploaded:
                file_paths = [os.path.join(downloads_folder, secure_filename(file.filename)) for file in files if allowed_file(file.filename)]
                messages, csv_only = csv_identification(file_paths, processed_data_keywords)
                session['csv_messages'] = messages
                session['file_path_feature']=file_path_feature
                print("file feature path:",file_path_feature)
                return render_template('success_and_redirect.html', message="All files successfully read", redirect_url=url_for('csv_files'))
            else:
                return render_template('alert.html', message="No CSV or Excel files to process", alert_type='warning')

    else:
        return render_template("upload.html")

@bp_file_reader.route('/csv_features', methods=['GET', 'POST'])
def csv_features_function():
    try:
        print("entered here")
        # Check if features file path is already stored in session
        files = session.get('file_path_feature', [])
        features_list = []
        print("feature file path else",files)
        # Define a directory to save the features file
        features_dir = os.path.join(os.getcwd(), 'features_files')
        print("entered here 2")

        if not os.path.exists(features_dir):
            print("entered here 2")
            os.makedirs(features_dir)
            print("entered here 3")

        

        for file_path in files:
            raw, sfreq= read_eeg_file(file_path)
            # This function should return a DataFrame
            csv_features_dataframe = csv_features(raw)
            features_list.append(csv_features_dataframe)
            print("Processing file csv in upload:", file_path)

        # Combine all feature DataFrames
        features = pd.concat(features_list, ignore_index=True)

        # Save the features DataFrame to a file
        features_file_name = secure_filename('features.csv')
        features_file_path = os.path.join(features_dir, features_file_name)
        features.to_csv(features_file_path, index=False)

        # Store the path to this file in the session
        session['features_file_path'] = features_file_path

        # Return a success response or render a template as needed
        return jsonify({'message': 'Features processed successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

        

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'edf', 'mat', 'xlsx', 'xlsm', 'xlsb','txt'}

@bp_file_reader.route('/calculate-svc', methods=['POST'])
def calculate_svc():
    try:
        # Get the first item from the list of label files
        labels_original = session.get('file_path_label_original', [None])[0]
        if labels_original is None:
            raise ValueError("No label conditions file found.")

        print("labels_original", labels_original)
        accuracies, results, labels_array_original = csv_svc_model(labels_original)
        session['labels_array_original'] = labels_array_original
        
        return jsonify({
            'accuracies': accuracies,
            'results': results,
        })
    
    except Exception as e:
        print(traceback.format_exc())  # This will print the full traceback to the console
        return jsonify({'error': str(e)}), 500
    
@bp_file_reader.route('/calculate-random', methods=['POST'])
def calculate_random():
    try:
        
        labels_original = session.get('file_path_label_original', [None])[0]
        if labels_original is None:
            raise ValueError("No label conditions file found.")

        print("labels_original", labels_original)
        accuracies, results, labels_array_original = csv_random_model(labels_original)
        session['labels_array_original'] = labels_array_original
        print("labels in random:",labels_array_original)
        return jsonify({
            'accuracies': accuracies,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500   

@bp_file_reader.route('/calculate-logistic', methods=['POST'])
def calculate_logistic():
    try:
        labels_original = session.get('file_path_label_original', [None])[0]
        if labels_original is None:
            raise ValueError("No label conditions file found.")

        print("labels_original", labels_original)
        accuracies, results, labels_array_original = csv_logistic_model(labels_original)
        session['labels_array_original'] = labels_array_original
        return jsonify({
            'accuracies': accuracies,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500    

@bp_file_reader.route('/calculate-knn', methods=['POST'])
def calculate_knn():
    try:
        labels_original = session.get('file_path_label_original', [None])[0]
        if labels_original is None:
            raise ValueError("No label conditions file found.")

        print("labels_original", labels_original)
        accuracies, results, labels_array_original = csv_knn_model(labels_original)
        session['labels_array_original'] = labels_array_original
        return jsonify({
            'accuracies': accuracies,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500  

@bp_file_reader.route('/calculate-cnn', methods=['POST'])
def calculate_cnn():
    try:
        labels_original = session.get('file_path_label_original', [None])[0]
        if labels_original is None:
            raise ValueError("No label conditions file found.")

        print("labels_original", labels_original)
        accuracies, results, labels_array_original = csv_cnn_model(labels_original)
        session['labels_array_original'] = labels_array_original
        return jsonify({
            'accuracies': accuracies,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500            
    
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
            label_get=[]
            labels_message=[]

            for file_path in mat_file_paths:
                print("Processing file paths:", mat_file_paths)
                filename = os.path.basename(file_path)
                filename_parts = filename.split('_')
                subject_identifier = '_'.join(filename_parts[1:3])  # sub-XXX_ses-XX

                raw, sfreq, labels = read_mat_eeg(file_path)
                preprocessed_raw, preprocessing_steps = preprocess_raw_eeg(raw, sfreq, subject_identifier)
                features_message,features_df = extract_features_csp(preprocessed_raw, 250, labels, epoch_length=1.0)
                accuracy_messages,label_get= mat_modeling_svc(subject_identifier, features_df, labels)
                labels_message.append(label_get)
                for step in preprocessing_steps:
                            if step not in all_preprocessing_steps:
                                all_preprocessing_steps.append(step)    
            
            session['labels_message']=labels_message
            print("labels message svc:",labels_message)
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
    

@bp_file_reader.route('/perform_mat_modeling_svc', methods=['POST'])
def perform_mat_modeling_svc():
    try:
        features_file_path = session.get('features_file_path')
        if not features_file_path or not os.path.exists(features_file_path):
            raise FileNotFoundError("Features file path not found or file does not exist.")

        # Read features_df from the CSV file
        features_df = pd.read_csv(features_file_path)
        subject_identifier = session.get('subject_identifier')
        labels = session.get('labels')


        # Call the mat_modeling function
        accuracy_messages,labels_message= mat_modeling_svc(subject_identifier, features_df, labels)

        # Store the accuracy messages in the session or pass to the template
        session['accuracy_messages'] = accuracy_messages
        

        return jsonify(accuracy=accuracy_messages)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
@bp_file_reader.route('/perform_mat_modeling_random', methods=['POST'])
def perform_mat_modeling_random():
    try:
        features_file_path = session.get('features_file_path')
        if not features_file_path or not os.path.exists(features_file_path):
            raise FileNotFoundError("Features file path not found or file does not exist.")

        # Read features_df from the CSV file
        features_df = pd.read_csv(features_file_path)
        subject_identifier = session.get('subject_identifier')
        labels = session.get('labels')


        # Call the mat_modeling function
        accuracy_messages,labels_message = mat_modeling_random(subject_identifier, features_df, labels)

        # Store the accuracy messages in the session or pass to the template
        session['accuracy_messages'] = accuracy_messages
        

        return jsonify(accuracy=accuracy_messages)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500   

@bp_file_reader.route('/perform_mat_modeling_logistic', methods=['POST'])
def perform_mat_modeling_logistic():
    try:
        features_file_path = session.get('features_file_path')
        if not features_file_path or not os.path.exists(features_file_path):
            raise FileNotFoundError("Features file path not found or file does not exist.")

        # Read features_df from the CSV file
        features_df = pd.read_csv(features_file_path)
        subject_identifier = session.get('subject_identifier')
        labels = session.get('labels')


        # Call the mat_modeling function
        accuracy_messages,labels_message = mat_modeling_logistic(subject_identifier, features_df, labels)

        # Store the accuracy messages in the session or pass to the template
        session['accuracy_messages'] = accuracy_messages
       
        

        return jsonify(accuracy=accuracy_messages)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500  

@bp_file_reader.route('/perform_mat_modeling_knn', methods=['POST'])
def perform_mat_modeling_knn():
    try:
        features_file_path = session.get('features_file_path')
        if not features_file_path or not os.path.exists(features_file_path):
            raise FileNotFoundError("Features file path not found or file does not exist.")

        # Read features_df from the CSV file
        features_df = pd.read_csv(features_file_path)
        subject_identifier = session.get('subject_identifier')
        labels = session.get('labels')


        # Call the mat_modeling function
        accuracy_messages,labels_message = mat_modeling_knn(subject_identifier, features_df, labels)

        # Store the accuracy messages in the session or pass to the template
        session['accuracy_messages'] = accuracy_messages
        

        return jsonify(accuracy=accuracy_messages)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500    

@bp_file_reader.route('/perform_mat_modeling_cnn', methods=['POST'])
def perform_mat_modeling_cnn():
    try:
        features_file_path = session.get('features_file_path')
        if not features_file_path or not os.path.exists(features_file_path):
            raise FileNotFoundError("Features file path not found or file does not exist.")

        # Read features_df from the CSV file
        features_df = pd.read_csv(features_file_path)
        subject_identifier = session.get('subject_identifier')
        labels = session.get('labels')


        # Call the mat_modeling function
        accuracy_messages,labels_message= mat_modeling_cnn(subject_identifier, features_df, labels)

        # Store the accuracy messages in the session or pass to the template
        session['accuracy_messages'] = accuracy_messages
        

        return jsonify(accuracy=accuracy_messages)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500       


def allowed_file_pred(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'xlsx', 'xlsm', 'xlsb','txt'}

@bp_file_reader.route('/predictions', methods=['GET', 'POST'])
def predictions():
    print("Entered prediction")
    file_path_csv=[]
    if request.method == 'POST':

        files = request.files.getlist('file_path_predict')

        if not files:
            return jsonify({'status': 'warning', 'message': "No files uploaded"}), 400
        
        # Assume we will process files successfully
        all_files_successfully = True
        message_details = []
        

        for file in files:
            if file.filename == '':
                all_files_successfully = False
                message_details.append("No selected files")
                continue

            if file and allowed_file_pred(file.filename):
                filename = secure_filename(file.filename)
                downloads_folder = os.path.expanduser("~/Downloads")
                file_path = os.path.join(downloads_folder, filename)
                file.save(file_path)

                extension = file.filename.rsplit('.', 1)[1].lower()

                try:
                    if extension in {'csv', 'xls', 'xlsx', 'xlsm', 'xlsb'}:
                        raw, sfreq = read_eeg_file(file_path)
                        if raw is not None:
                            file_path_csv.append(file_path)
                    elif extension in {'txt'}:
                        label_conditions = read_label_conditions(file_path)  # Updated to use the existing function
                        session['label_conditions']=label_conditions
                    else:
                        raise ValueError("Unsupported file type")
                except ValueError as e:
                    all_files_successfully = False
                    message_details.append(str(e))
                except Exception as e:
                    all_files_successfully = False
                    message_details.append(f"An error occurred while processing {filename}: {e}")
            else:
                all_files_successfully = False
                message_details.append(f"Unsupported file type: {file.filename}")

        if all_files_successfully:
            print("file path csv:",file_path_csv)
            return jsonify({'status': 'success', 'message': "All files processed successfully.", 'file_path_csv': file_path_csv})

        else:
            error_message = " ".join(message_details)
            logging.error(f'Error processing files: {error_message}')
            print(traceback.format_exc())
            return jsonify({'status': 'error', 'message': error_message}), 400
    else:
        return render_template("predictions.html")
    

@bp_file_reader.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data.get('model')
    file_path_csv = data.get('filenames', [])
    features_list = []
    print("Processing file paths:", file_path_csv)

    
    for file_path in file_path_csv:
        raw, sfreq= read_eeg_file(file_path)
        csv_features_dataframe = csv_features(raw)
        if 'label' in csv_features_dataframe.columns:
            csv_features_dataframe.drop('label', axis=1, inplace=True)
        features_list.append(csv_features_dataframe)
        print("Processing file csv:", file_path)

    features = pd.concat(features_list, ignore_index=True)
    print("Features:", features)
    label_conditions = session.get('label_conditions', {})
    label_messages_svc=session.get('labels_array_original')
    print("label conditions", label_conditions)

    try:
        
        if model_name == 'svc':
            predictions = load_and_predict_svc(features, label_conditions)
        elif model_name == 'random':
            predictions = load_and_predict_random(features, label_conditions)
        elif model_name == 'knn':
            predictions = load_and_predict_knn(features, label_conditions)
        elif model_name == 'cnn':
            predictions = load_and_predict_cnn(features, label_conditions)
        elif model_name == 'logistic':
            predictions = load_and_predict_logisitc(features, label_conditions)
        else:
            return jsonify({'error': 'Invalid model name'}), 400
        
        comparison_results = []

        # Define the models you want to iterate over
        models = ['Model_RF', 'Model_SVC', 'Model_LR','Model_KNN','Model_CNN']

        # Iterate over each model
        for model_key in models:
            if predictions is not None and model_key in predictions and label_messages_svc is not None:
                # Extract just the emotions from predictions for comparison
                predicted_emotions = [pred.split(' = ')[-1] for pred in predictions[model_key]]

                # Ensure we only compare lists of the same length
                min_length = min(len(predicted_emotions), len(label_messages_svc))
                comparison_results = ['Right' if predicted_emotions[i] == label_messages_svc[i] else 'Wrong' for i in range(min_length)]

                print("Predictions with comparison:", comparison_results)
                print("Predictions:", predictions)

                # Prepare the final list for response
                final_predictions_with_comparison = []

                # Keep the original 'Person x =' format and append comparison results
                for i, (pred, orig_label) in enumerate(zip(predictions[model_key], label_messages_svc)):
                    comparison_result = 'Right' if pred.split(' = ')[-1] == orig_label else 'Wrong'
                    final_predictions_with_comparison.append(f"{pred} - {comparison_result}")

                print("Predictions with comparison:", final_predictions_with_comparison)

                # Add the comparison results to the JSON response
                return jsonify({'predictions_with_comparison': final_predictions_with_comparison})
                
    except Exception as e:
        print(traceback.format_exc())
    # Return a more detailed error message to the client
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500    
    
@bp_file_reader.route('/mat_predict', methods=['GET', 'POST'])
def mat_predict():
    print("Entered mat predict")
    session.pop('csv_messages', None)
    

    if request.method == "POST":
        files = request.files.getlist('file_path_upload')
        mat_predict_file_paths = []
       

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
                print(f"File saved to {file_path}")

                # Determine the file type and process accordingly
                extension = file.filename.rsplit('.', 1)[1].lower()
                try:
                    if extension == 'mat':
                        mat_predict_file_paths.append(file_path)
                    else:
                        raise ValueError("Unsupported file type")
                    
                    if extension != 'mat':
                        raise ValueError(f"Failed to read file {filename}")
                
                except ValueError as e:
                    os.remove(file_path)
                    all_files_processed_successfully = False
                    return render_template('alert.html', message=str(e), alert_type='danger')
            else:
                all_files_processed_successfully = False
                return render_template('alert.html', message="Unsupported file type", alert_type='danger')

        if mat_predict_file_paths:
            session['mat_predict_file_paths'] = mat_predict_file_paths
            print("MAT file paths stored in session:", mat_predict_file_paths)

            if all_files_processed_successfully:
                # Return a JSON response indicating success
                return jsonify({'success': 'All files successfully read', 
                                'mat_predict_file_paths': mat_predict_file_paths})
            else:
                # Return a JSON response indicating an error
                return jsonify({'error': 'An error occurred while processing files.'}), 400
                
        else:
            return jsonify({'error': 'No MAT files detected or unsupported filetype.'}), 400

        
        
    return render_template("mat_predict.html")
    
@bp_file_reader.route('/mat_preprocess_predict', methods=['GET', 'POST'])      
def mat_preprocess_predict():

    if request.method=='POST':
        all_preprocessing_steps_predict = []

        try:

            mat_predict_file_paths = session.get('mat_predict_file_paths', [])
            
            

            for file_path in mat_predict_file_paths:
                print("Processing file paths:", mat_predict_file_paths)
                filename = os.path.basename(file_path)
                filename_parts = filename.split('_')
                subject_identifier = '_'.join(filename_parts[1:3])  # sub-XXX_ses-XX

                raw, sfreq, labels = read_mat_eeg(file_path)
                preprocessed_raw_predict, preprocessing_steps_predict = preprocess_raw_eeg(raw, sfreq, subject_identifier)
                for step in preprocessing_steps_predict:
                            if step not in all_preprocessing_steps_predict:
                                all_preprocessing_steps_predict.append(step)    
            
            
        
            # Convert labels to list if it's an ndarray
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()

            # Store labels in session
            some_directory = "E:\\EEG-Classification"  # Ensure this directory exists and has write permissions
            raw_file_path_predict = os.path.join(some_directory, f"{subject_identifier}_preprocessed_predict.fif")
            preprocessed_raw_predict.save(raw_file_path_predict, overwrite=True)

            # Store the reference to this file in the session
            session['preprocessed_predict_file_path'] = raw_file_path_predict
            session['subject_identifier'] = subject_identifier
            session['labels'] = labels
            # Clear the stored file paths after processing
            #session.pop('mat_predict_file_paths', None)
            # Retrieve preprocessing steps from session
            #preprocessing_steps = session.get('preprocessing_steps', [])
            # Store the collected preprocessing steps in the session
            session['preprocessing_steps_predict'] = all_preprocessing_steps_predict  # Store in session

            # Retrieve preprocessing steps from session, not from a local variable
            preprocessing_steps_predict = session.get('preprocessing_steps_predict', [])  # Retrieve from session

            # Function to extract the session number for sorting
            def get_session_number(step_description):
                # This assumes the format "Interpolated bad channels for sub-XXX_ses-XX"
                parts = step_description.split('_ses-')
                session_part = parts[1]
                session_number = int(session_part.split(':')[0])  # Convert to integer for correct numeric sorting
                return session_number

            # Reorder the steps
            ica_filter_steps = [step for step in preprocessing_steps_predict if "High-pass filtered" in step or "Applied ICA" in step]
            
            # Sort interpolated steps by extracting the session number
            interpolated_steps = sorted(
                [step for step in preprocessing_steps_predict if step.startswith("Interpolated bad channels for")],
                key=get_session_number
            )

            other_steps = [step for step in preprocessing_steps_predict if step not in ica_filter_steps and step not in interpolated_steps]

            # Concatenate the lists to get the desired order
            ordered_preprocessing_steps_predict = ica_filter_steps + other_steps + interpolated_steps

            return jsonify(preprocessing_steps_predict=ordered_preprocessing_steps_predict)
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@bp_file_reader.route('/extract-csp-features_predict', methods=['POST'])
def extract_csp_features_predict():
    try:
        # Retrieve preprocessed data and other necessary variables from session
        raw_file_path = session.get('preprocessed_predict_file_path')
        if not raw_file_path or not os.path.exists(raw_file_path):
            raise FileNotFoundError("Preprocessed raw file path not found or file does not exist.")

        # Load the preprocessed Raw object from the file
        preprocessed_raw = mne.io.read_raw_fif(raw_file_path, preload=True)
        subject_identifier = session.get('subject_identifier')
        labels = session.get('labels')

        # Extract CSP features
        features_message,features_df = extract_features_csp(preprocessed_raw, 250, labels, epoch_length=1.0)

        # Save features_df to a temporary file
        _, temp_file_path_predict = tempfile.mkstemp(suffix='.csv')  # Creates a temp file
        features_df.to_csv(temp_file_path_predict, index=False)  # Save DataFrame to CSV

        # Store the path of the temporary file in the session
        session['features_file_path_predict'] = temp_file_path_predict

        # Convert DataFrame to JSON
        features_json = features_df.to_json(orient='records')
        

        return jsonify(features=features_json, message=features_message)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@bp_file_reader.route('/perform_mat_prediction', methods=['POST'])
def perform_mat_prediction():
    try:
        data = request.get_json()
        mat_predict_file_paths = session.get('mat_predict_file_paths', [])
        model_name = data.get('model_name') 
        print(f"Model name from session: {model_name}")  # Debugging print

        label_messages=session.get('labels_message')
        print("labels message in predict:",label_messages)

        all_subjects_predictions = []  # List to store predictions for all subjects

        # Define a dictionary to map model names to prediction functions
        prediction_functions = {
            'svc': predict_movement_svc,
            'random': predict_movement_random,
            'logistic': predict_movement_logistic,
            'knn': predict_movement_knn,
            'cnn': predict_movement_cnn,
        }

        # Check if the specified model_name has a corresponding prediction function
        if model_name not in prediction_functions:
            raise ValueError(f"No prediction function available for the model '{model_name}'")

        # Get the appropriate prediction function
        predict_function = prediction_functions[model_name]

        for file_path in mat_predict_file_paths:
            print("Processing file paths:", mat_predict_file_paths)
            filename = os.path.basename(file_path)
            filename_parts = filename.split('_')
            subject_identifier = '_'.join(filename_parts[1:3])  # sub-XXX_ses-XX

            raw, sfreq, labels = read_mat_eeg(file_path)
            preprocessed_raw_predict, preprocessing_steps_predict = preprocess_raw_eeg(raw, sfreq, subject_identifier)
            features_message, features_df = extract_features_csp(preprocessed_raw_predict, sfreq, labels)
            
            # Call the appropriate predict function
            subject_predictions = predict_function(features_df, subject_identifier)
            all_subjects_predictions.extend(subject_predictions)  # Add the result to the list

        # Store the accuracy messages in the session or pass to the template
        session['predict_message'] = all_subjects_predictions
        session.pop('mat_predict_file_paths', None)  # Clear the file paths to prevent reprocessing

        return jsonify(predict=all_subjects_predictions, label_messages=session.get('labels_message'))
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
 