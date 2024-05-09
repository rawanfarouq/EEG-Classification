from flask import Blueprint, render_template, request, jsonify, flash,redirect,url_for, session,send_file
from werkzeug.utils import secure_filename
from collections import Counter
from sklearn.metrics import confusion_matrix
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import blue, red, green, purple,black,blueviolet
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate,Spacer , Table,TableStyle
from reportlab.lib import colors
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing
import sys
import os
import io
import mne
import numpy as np
import tempfile
import pandas as pd 
import traceback, shutil,tempfile, logging, json


bp_file_reader = Blueprint('file_reader', __name__)

# Calculate the absolute path to the backend directory
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
print("Backend directory:", backend_dir)

# Add the backend directory to the sys.path
sys.path.insert(0, backend_dir)

from backend.classification import read_eeg_file, read_edf_eeg, read_mat_eeg, csv_identification,processed_data_keywords,csv_svc_model,preprocess_raw_eeg,extract_features_csp,mat_modeling,get_label_text,read_label_conditions,train_test_split_models
from backend.classification import csv_svc_model_new,csv_random_model,csv_logistic_model,csv_knn_model,csv_cnn_model,load_and_predict_svc,load_and_predict_random,load_and_predict_knn,load_and_predict_cnn,load_and_predict_logisitc,csv_features,predict_on_training_data
from backend.classification import mat_modeling_svc,mat_modeling_random,mat_modeling_logistic,mat_modeling_knn,mat_modeling_cnn,predict_movement,predict_movement_svc,predict_movement_random,predict_movement_logistic,predict_movement_knn,predict_movement_cnn
from backend.features import extract_features


@bp_file_reader.route('/upload', methods=['GET', 'POST'])
def upload():
    print("Entered upload")
    session.pop('csv_messages', None)
    file_path_feature=[]
    csv_file_path=[]
    session['model_accuracies'] = {}
    # Initialize the accuracies dictionary if it doesn't exist
    if 'model_accuracies' not in session:
        session['model_accuracies'] = {}

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
                        csv_file_path.append(file_path)
                        # print(f"Calling read_eeg_file for {file_path}")
                        # raw, sfreq = read_eeg_file(file_path)
                        # print(f"read_eeg_file returned {len(raw)} samples for {file_path}")
                        
                    elif extension =='txt':
                        file_path_label_original.append(file_path)
                        session['file_path_label_original']=file_path_label_original
                        label_conditions_in_predict_train = read_label_conditions(file_path)
                        session['label_conditions_in_predict_train']=label_conditions_in_predict_train
                        print("label_conditions_in_predict_train",label_conditions_in_predict_train)

                    elif extension == 'mat':
                        mat_file_paths.append(file_path)
                    else:
                        raise ValueError("Unsupported file type")
                    
                    if extension != 'mat'and  extension not in {'csv', 'xls', 'xlsx', 'xlsm', 'xlsb'} and extension!='txt':
                        raise ValueError(f"Failed to read file {filename}")
                
                except ValueError as e:
                    os.remove(file_path)
                    all_files_processed_successfully = False
                    return render_template('alert.html', message=str(e), alert_type='danger')
            else:
                all_files_processed_successfully = False
                return render_template('alert.html', message="Unsupported file type", alert_type='danger')

        if mat_file_paths:
            #extract_features(mat_file_paths)
            session['mat_file_paths'] = mat_file_paths
            print("MAT file paths stored in session:", mat_file_paths)
            # Redirect to mat_classification only if mat files are uploaded and no other types are present
            if not csv_or_excel_uploaded:
                return render_template('success_and_redirect.html', message="All files successfully read", redirect_url=url_for('mat_classification'))
        
        if all_files_processed_successfully:
            # Redirect to csv_files if any CSV or Excel files were uploaded
            if csv_or_excel_uploaded:
                file_paths = [os.path.join(downloads_folder, secure_filename(file.filename)) for file in files if allowed_file(file.filename)]
                # messages, csv_only = csv_identification(file_paths, processed_data_keywords)
                # session['csv_messages'] = messages
                session['csv_file_path']=csv_file_path
                session['file_path_feature']=file_path_feature
                print("file feature path:",file_path_feature)
                return render_template('success_and_redirect.html', message="All files successfully read", redirect_url=url_for('csv_files'))
            else:
                return render_template('alert.html', message="No CSV or Excel files to process", alert_type='warning')

    else:
        return render_template("upload.html")
    


@bp_file_reader.route('/check_csv', methods=['GET', 'POST'])
def check_csv():
    try:

        files = session.get('csv_file_path', [])
        print("file identify:", files)

        results = []

        for file_path in files:
            raw, sfreq, cleanliness_messages = read_eeg_file(file_path)
            
            # Construct a dictionary for the response
            result = {
                'cleanliness_messages': cleanliness_messages
            }
            results.append(result)
            print("results:",results)

        # Return a JSON response containing all results
        return jsonify(results)
    except Exception as e:
        print(traceback.format_exc())  # This will print the full traceback to the console

        return jsonify({'error': str(e)}), 500

@bp_file_reader.route('/csv_identify', methods=['GET', 'POST'])
def csv_identify():
    try:
        # Check if features file path is already stored in session
        files = session.get('csv_file_path', [])
        print("file identify:",files)
        
        
        messages, csv_only = csv_identification(files, processed_data_keywords)
        print("Processing file csv in upload:", files)

        # Return a success response or render a template as needed
        return jsonify(messages=messages), 200

    except Exception as e:
        print(traceback.format_exc())  # This will print the full traceback to the console

        return jsonify({'error': str(e)}), 500


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
            raw, sfreq,clean_message= read_eeg_file(file_path)
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
        accuracies, results, labels_array_original,y_original = csv_svc_model_new(labels_original)

        session['y_original']=y_original

        accuracy_value = results['Accuracy']
        if isinstance(accuracy_value, list):
            # If the accuracy is a list, take the first element or perform other logic to get a single value
            accuracy_value = accuracy_value[0]
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['SVC'] = accuracy_value

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
        accuracies, results, labels_array_original,y_original = csv_random_model(labels_original)

        session['y_original']=y_original

        accuracy_value = results['Accuracy']
        if isinstance(accuracy_value, list):
            # If the accuracy is a list, take the first element or perform other logic to get a single value
            accuracy_value = accuracy_value[0]
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['Random'] = accuracy_value

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
        accuracies, results, labels_array_original,y_original = csv_logistic_model(labels_original)

        session['y_original']=y_original

        accuracy_value = results['Accuracy']
        if isinstance(accuracy_value, list):
            # If the accuracy is a list, take the first element or perform other logic to get a single value
            accuracy_value = accuracy_value[0]
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['Logistic'] = accuracy_value
 

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
        accuracies, results, labels_array_original,y_original = csv_knn_model(labels_original)

        session['y_original']=y_original

        accuracy_value = results['Accuracy']
        if isinstance(accuracy_value, list):
            # If the accuracy is a list, take the first element or perform other logic to get a single value
            accuracy_value = accuracy_value[0]
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['KNN'] = accuracy_value


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
        accuracies, results, labels_array_original,y_original = csv_cnn_model(labels_original)

        session['y_original']=y_original

        accuracy_value = results['Accuracy']
        if isinstance(accuracy_value, list):
            # If the accuracy is a list, take the first element or perform other logic to get a single value
            accuracy_value = accuracy_value[0]
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['CNN'] = accuracy_value


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
            features_csv_paths = []
            mat_file_paths = session.get('mat_file_paths', [])
            print("mat file path:",mat_file_paths)
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
                some_directory = "E:\\EEG-Classification"
                if not os.path.exists(some_directory):
                    # If it doesn't exist, use the Downloads folder as a fallback
                    home_dir = os.path.expanduser("~")  # Get the home directory
                    some_directory = os.path.join(home_dir, 'Downloads')

                # Make sure the fallback directory exists or create it
                os.makedirs(some_directory, exist_ok=True)
                csv_path = os.path.join(some_directory, f"{subject_identifier}_features.csv")
                features_df.to_csv(csv_path, index=False)
                features_csv_paths.append((subject_identifier, csv_path))
                for step in preprocessing_steps:
                            if step not in all_preprocessing_steps:
                                all_preprocessing_steps.append(step)    
            
            session['labels_message']=labels_message
            print("labels message svc:",labels_message)
            session['features_csv_paths'] = features_csv_paths

            # Convert labels to list if it's an ndarray
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()

            # Store labels in session
            session['labels'] = labels
            some_directory = "E:\\EEG-Classification"  # Ensure this directory exists and has write permissions
            if not os.path.exists(some_directory):
                # If it doesn't exist, use the Downloads folder as a fallback
                home_dir = os.path.expanduser("~")  # Get the home directory
                some_directory = os.path.join(home_dir, 'Downloads')

            # Make sure the fallback directory exists or create it
            os.makedirs(some_directory, exist_ok=True)
            raw_file_path = os.path.join(some_directory, f"{subject_identifier}_preprocessed_raw.fif")
            preprocessed_raw.save(raw_file_path, overwrite=True)

            # Store the reference to this file in the session
            session['preprocessed_raw_file_path'] = raw_file_path
            session['subject_identifier'] = subject_identifier
            session['labels'] = labels
            # Clear the stored file paths after processing
            #session.pop('mat_file_paths', None)
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
            session['download_preprocess_steps']=ordered_preprocessing_steps

            return jsonify(preprocessing_steps=ordered_preprocessing_steps)
        except Exception as e:
            print(traceback.format_exc())  # This will print the full traceback to the console

            return jsonify({'error': str(e)}), 500
    else:         
        return render_template('mat_files_class.html')
    
@bp_file_reader.route('/extract-csp-features', methods=['POST'])
def extract_csp_features():
    try:
        session['download_extract_message']=[]
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

        session['download_extract_message']=features_message
        

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

        accuracy_value = None
        for message in accuracy_messages:
            if 'Accuracy' in message:
                # Extract the accuracy percentage from the string
                accuracy_str = message.split('Accuracy: ')[1].split('%')[0]
                try:
                    accuracy_value = float(accuracy_str)
                except ValueError:
                    # Handle the exception if the conversion fails
                    print(f"Could not convert accuracy to float: '{accuracy_str}'")
                break  # Assuming we only need the first occurrence

        if accuracy_value is None:
            raise ValueError("Accuracy value was not found in accuracy messages.")
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['SVC'] = accuracy_value

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

        accuracy_value = None
        for message in accuracy_messages:
            if 'Accuracy' in message:
                # Extract the accuracy percentage from the string
                accuracy_str = message.split('Accuracy: ')[1].split('%')[0]
                try:
                    accuracy_value = float(accuracy_str)
                except ValueError:
                    # Handle the exception if the conversion fails
                    print(f"Could not convert accuracy to float: '{accuracy_str}'")
                break  # Assuming we only need the first occurrence

        if accuracy_value is None:
            raise ValueError("Accuracy value was not found in accuracy messages.")
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['Random'] = accuracy_value

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

        accuracy_value = None
        for message in accuracy_messages:
            if 'Accuracy' in message:
                # Extract the accuracy percentage from the string
                accuracy_str = message.split('Accuracy: ')[1].split('%')[0]
                try:
                    accuracy_value = float(accuracy_str)
                except ValueError:
                    # Handle the exception if the conversion fails
                    print(f"Could not convert accuracy to float: '{accuracy_str}'")
                break  # Assuming we only need the first occurrence

        if accuracy_value is None:
            raise ValueError("Accuracy value was not found in accuracy messages.")
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['Logistic'] = accuracy_value

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

        accuracy_value = None
        for message in accuracy_messages:
            if 'Accuracy' in message:
                # Extract the accuracy percentage from the string
                accuracy_str = message.split('Accuracy: ')[1].split('%')[0]
                try:
                    accuracy_value = float(accuracy_str)
                except ValueError:
                    # Handle the exception if the conversion fails
                    print(f"Could not convert accuracy to float: '{accuracy_str}'")
                break  # Assuming we only need the first occurrence

        if accuracy_value is None:
            raise ValueError("Accuracy value was not found in accuracy messages.")
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['KNN'] = accuracy_value

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

        accuracy_value = None
        for message in accuracy_messages:
            if 'Accuracy' in message:
                # Extract the accuracy percentage from the string
                accuracy_str = message.split('Accuracy: ')[1].split('%')[0]
                try:
                    accuracy_value = float(accuracy_str)
                except ValueError:
                    # Handle the exception if the conversion fails
                    print(f"Could not convert accuracy to float: '{accuracy_str}'")
                break  # Assuming we only need the first occurrence

        if accuracy_value is None:
            raise ValueError("Accuracy value was not found in accuracy messages.")
        #Store the accuracy, so you can get the highest accuracy
        session['model_accuracies']['CNN'] = accuracy_value

        # Store the accuracy messages in the session or pass to the template
        session['accuracy_messages'] = accuracy_messages
        

        return jsonify(accuracy=accuracy_messages)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500 
    
    
@bp_file_reader.route('/get_highest_accuracy', methods=['GET'])
def get_highest_accuracy():
    if 'model_accuracies' in session and session['model_accuracies']:
        # Convert the stored accuracies to float for comparison
        accuracies_float = {model: float(acc) for model, acc in session['model_accuracies'].items()}
        highest_accuracy_model = max(accuracies_float, key=accuracies_float.get)
        highest_accuracy = accuracies_float[highest_accuracy_model]

        return jsonify(highest_accuracy_model=highest_accuracy_model, highest_accuracy=highest_accuracy)
    else:
        return jsonify(error="No model accuracies found. Please select a model"), 404


@bp_file_reader.route('/perform_train_test_mat', methods=['POST'])
def perform_train_test_mat():
    try:
        session['mat_model']=[]
        session['mat_prediction']=[]
        # Parse the JSON request data
        data = request.get_json()
        model_name = data.get('model')
        combined_predictions = []
        session['mat_model']=model_name

        # Retrieve the list of feature CSV paths and subject identifiers from the session
        features_csv_paths = session.get('features_csv_paths')


        for subject_identifier,features_csv_path in features_csv_paths:
            print("subject iden:",subject_identifier)
            features_df = pd.read_csv(features_csv_path)
            labels = session.get('labels')

            # Call the train_test_split_models function for each subject
            subject_predictions, subject_metrics = train_test_split_models(
                subject_identifier, features_df, labels, model_name, test_size=0.2
            )

            # Function to map label numbers to descriptions
            def map_label_to_description(label):
                if label == 1:
                    return "Moving Left Hand"
                else:
                    return "Moving Right Hand"

            # Map labels to descriptions and sanitize int32
            sanitized_predictions = [
                {
                    k: (map_label_to_description(v) if k in ['actual_label', 'predicted_label'] else int(v))
                    if isinstance(v, np.int32) else v
                    for k, v in prediction.items()
                }
                for prediction in subject_predictions # Adjusted to access 'predictions'
            ]

            # Add the metrics to the combined_predictions dictionary
            sanitized_predictions_with_metrics = {
                'predictions': sanitized_predictions,
                'metrics': subject_metrics
            }

            # Add the sanitized predictions and metrics for the subject to the combined list
            combined_predictions.append(sanitized_predictions_with_metrics)

        session['mat_prediction']=combined_predictions
        # Modify the return statement to include metrics
        return jsonify({
            'status': 'success',
            'all_subjects_predictions_with_metrics': combined_predictions,
        })

    except ValueError as e:
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'details': str(e)}), 400
    except Exception as e:
        # Handle other exceptions
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'details': str(e)}), 500
    

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
                        raw, sfreq,clean_message = read_eeg_file(file_path)
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
    session['model_name']=[]
    session['predictions_with_comparison']=[]
    session['result_predict']=[]
    data = request.get_json()
    model_name = data.get('model')
    session['model_name']=model_name
    file_path_csv = data.get('filenames', [])
    features_list = []
    print("Processing file paths:", file_path_csv)

    
    for file_path in file_path_csv:
        raw, sfreq,clean_message= read_eeg_file(file_path)
        csv_features_dataframe = csv_features(raw)
        if 'label' in csv_features_dataframe.columns:
            csv_features_dataframe.drop('label', axis=1, inplace=True)
        features_list.append(csv_features_dataframe)
        print("Processing file csv:", file_path)

    features = pd.concat(features_list, ignore_index=True)
    print("Features:", features)
    label_conditions = session.get('label_conditions', {})
    label_messages_svc=session.get('labels_array_original')
    true_labels_original=session.get('y_original',[])
    print("label conditions", label_conditions)

    try:
        
        if model_name == 'svc':
            predictions= load_and_predict_svc(features, label_conditions,true_labels_original)
        elif model_name == 'random':
            predictions= load_and_predict_random(features, label_conditions,true_labels_original)
        elif model_name == 'knn':
            predictions = load_and_predict_knn(features, label_conditions,true_labels_original)
        elif model_name == 'cnn':
            predictions = load_and_predict_cnn(features, label_conditions,true_labels_original)
        elif model_name == 'logistic':
            predictions = load_and_predict_logisitc(features, label_conditions,true_labels_original)
        else:
            return jsonify({'error': 'Invalid model name'}), 400
        
        comparison_results = []
        result_predict = {'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': [], 'Confusion Matrix':[]}
        
     


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

                # Calculate the overall accuracy
                accuracy = comparison_results.count('Right') / len(comparison_results) * 100

                # Prepare the final list for response
                final_predictions_with_comparison = []

                # Keep the original 'Person x =' format and append comparison results
                for i, (pred, orig_label) in enumerate(zip(predictions[model_key], label_messages_svc)):
                    comparison_result = 'Right' if pred.split(' = ')[-1] == orig_label else 'Wrong'
                    final_predictions_with_comparison.append(f"{pred} - {comparison_result}")

                print("Predictions with comparison:", final_predictions_with_comparison)

                        # Fetch the true labels and the predicted labels
                true_labels = label_messages_svc  # These should be the original true labels
                predicted_labels = [pred.split(' = ')[-1] for pred in predictions[model_key]]  # Assuming Model_SVC is being used here
                
                # Ensure we only compare lists of the same length
                min_length = min(len(predicted_labels), len(true_labels))
                true_labels = true_labels[:min_length]
                predicted_labels = predicted_labels[:min_length]
                
                # Calculate comparison results
                comparison_results = ['Right' if true_labels[i] == predicted_labels[i] else 'Wrong' for i in range(min_length)]

                # Calculate the overall accuracy
                accuracy = comparison_results.count('Right') / len(comparison_results) * 100

                # Count occurrences of each class in the true labels
                true_label_counts = Counter(true_labels)

                # Initialize counts for true positives, false positives, and false negatives
                tp = Counter()
                fp = Counter()
                fn = Counter()

                # Calculate tp, fp, fn for each class
                for i, label in enumerate(true_labels):
                    if label == predicted_labels[i]:
                        tp[label] += 1
                    else:
                        fp[predicted_labels[i]] += 1
                        fn[label] += 1
                
                # Calculate precision, recall, and F1 score for each class
                precision_per_class = {label: tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0 for label in true_label_counts}
                recall_per_class = {label: tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0 for label in true_label_counts}
                f1_per_class = {label: 2 * (precision_per_class[label] * recall_per_class[label]) / (precision_per_class[label] + recall_per_class[label]) if (precision_per_class[label] + recall_per_class[label]) > 0 else 0 for label in true_label_counts}
                
                # Calculate macro-averaged precision, recall, and F1 score
                macro_precision = sum(precision_per_class.values()) / len(precision_per_class)*100
                macro_recall = sum(recall_per_class.values()) / len(recall_per_class)*100
                macro_f1 = sum(f1_per_class.values()) / len(f1_per_class)*100

                # Calculate the confusion matrix
                labels_unique = sorted(set(true_labels + predicted_labels))  # Sorted list of unique labels
                conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels_unique)
                
                # Convert the numpy array to a list of lists for JSON serialization
                conf_matrix_list = conf_matrix.tolist()

                print(f"Overall accuracy: {accuracy}%")
                print(f"Overall precision: {macro_precision}%")
                print(f"Overall recall: {macro_recall}%")
                print(f"Overall f1 score: {macro_f1}%")
                print(f"confusion matrix: {conf_matrix}")



        result_predict['Accuracy'].append(accuracy)
        result_predict['Precision'].append(macro_precision)        
        result_predict['Recall'].append(macro_recall)
        result_predict['F1 Score'].append(macro_f1)
        result_predict['Confusion Matrix'].append(conf_matrix_list)

        session['predictions_with_comparison']=final_predictions_with_comparison
        session['result_predict']=result_predict


        # Add the comparison results to the JSON response
        return jsonify({
            'predictions_with_comparison': final_predictions_with_comparison,
            'result_predict': result_predict
        })
                
    except Exception as e:
        print(traceback.format_exc())
    # Return a more detailed error message to the client
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500   



def numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: numpy_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [numpy_to_list(item) for item in data]
    else:
        return data

@bp_file_reader.route('/perform_predict_on_training_data', methods=['POST'])
def perform_predict_on_training_data():
    session['model_name']=[]
    session['predictions_with_comparison']=[]
    session['download_performance']=[]
    session['result_predict']=[]
    data = request.get_json()
    model_name = data.get('model')
    session['model_name']=model_name

    try:
        label_conditions = session.get('label_conditions_in_predict_train', {})
        print("label condition in predict train:", label_conditions)
        formatted_predictions_train, result_predict_train = predict_on_training_data(model_name, label_conditions)

        # Convert all numpy arrays in the results to lists
        result_predict_train = numpy_to_list(result_predict_train)
        session['predictions_with_comparison']=formatted_predictions_train
        session['result_predict']=result_predict_train
        session['download_performance']=result_predict_train

        return jsonify({
            'status': 'success',
            'formatted_predictions_train': formatted_predictions_train,
            'result_predict_train': result_predict_train
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'details': str(e)}), 500
    
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
            if not os.path.exists(some_directory):
                # If it doesn't exist, use the Downloads folder as a fallback
                home_dir = os.path.expanduser("~")  # Get the home directory
                some_directory = os.path.join(home_dir, 'Downloads')

            # Make sure the fallback directory exists or create it
            os.makedirs(some_directory, exist_ok=True)
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
        session['mat_model']=[]
        session['mat_prediction']=[]
        session['download_performance']=[]
        data = request.get_json()
        mat_predict_file_paths = session.get('mat_predict_file_paths', [])
        model_name = data.get('model_name') 
        session['mat_model']=model_name
        print(f"Model name from session: {model_name}")  # Debugging print

        label_messages=session.get('labels_message')
        print("labels message in predict:",label_messages)

        all_subjects_predictions = []  # List to store predictions for all subjects
        all_subjects_metrics = {}  # Dictionary to store metrics for all subjects


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
            subject_predictions, subject_metrics = predict_function(features_df, subject_identifier)
            all_subjects_predictions.extend(subject_predictions)  # Add the result to the list
            all_subjects_metrics[subject_identifier] = subject_metrics  # Add metrics to the dictionary

        # Store the accuracy messages in the session or pass to the template
        session['predict_message'] = all_subjects_predictions
        #session.pop('mat_predict_file_paths', None)  # Clear the file paths to prevent reprocessing
        session['mat_prediction']=all_subjects_predictions

        session['download_performance']=all_subjects_metrics

        return jsonify(predict=all_subjects_predictions, label_messages=session.get('labels_message'), metrics=all_subjects_metrics)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    

def add_footer(canvas, doc):
    canvas.saveState()
    footer_text = "EEG Hub Analysis - Confidential Report"
    canvas.setFont("Helvetica", 9)
    canvas.drawString(40, 30, footer_text)
    canvas.restoreState()
    

@bp_file_reader.route('/download_predictions', methods=['GET'])
def download_predictions():
    try:
        predictions_with_comparison = session.get('predictions_with_comparison', [])
        result_predict = session.get('result_predict', {})
        selected_model = session.get('model_name')
        
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('title_style',
                                     parent=styles['Title'],
                                     fontName='Helvetica-Bold',
                                     fontSize=18,
                                     spaceAfter=12,
                                     textColor=black)
        heading_style = ParagraphStyle('heading_style',
                                       parent=styles['Heading2'],
                                       fontName='Helvetica-Bold',
                                       fontSize=14,
                                       spaceAfter=6,
                                       textColor=black)
        body_style = ParagraphStyle('body_style',
                                    parent=styles['BodyText'],
                                    fontName='Helvetica',
                                    fontSize=12,
                                    leading=15,
                                    textColor=black)

        # Title
        elements.append(Paragraph("EEG Classification Report", title_style))

        # Subtitle and Introduction
        intro_text = f"<b>Report generated by:</b> EEG Hub Analysis System<br/>"\
                     f"<b>Classification model used:</b> <font color='blue'>{selected_model}</font><br/><br/>"\
                     "This document provides a detailed overview of the EEG classification results, "\
                     "including predictions, comparisons against actual data, and key performance metrics."
        elements.append(Paragraph(intro_text, body_style))

        # Methodology Section
        methodology_text = "The classification was performed using a machine learning model that was "\
                           f"{selected_model}, trained and validated on a comprehensive dataset of EEG readings."
        elements.append(Paragraph("<b>Methodology:</b>", heading_style))
        elements.append(Paragraph(methodology_text, body_style))

        # Predictions
        elements.append(Paragraph("<b>Predictions:</b>", heading_style))
        for pred in predictions_with_comparison:
            elements.append(Paragraph(pred, body_style))

        # Metrics Section
        elements.append(Paragraph("<b>Performance Metrics:</b>", heading_style))
        for k, v in result_predict.items():
            metric_text = f"<b>{k}:</b> {v}"
            elements.append(Paragraph(metric_text, body_style))

        # Build PDF
        doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)

        # Move to the beginning of the BytesIO buffer
        pdf_buffer.seek(0)
        return send_file(pdf_buffer, as_attachment=True, download_name="EEG_Classification_Report.pdf", mimetype='application/pdf')

    except Exception as e:
        print(traceback.format_exc())  # This will print the full traceback to the console
        return jsonify({'error': str(e)}), 500
    

@bp_file_reader.route('/download_predictions_mat', methods=['GET'])
def download_predictions_mat():
    try:
        preprocess_steps = session.get('download_preprocess_steps', [])
        extract_messages = session.get('download_extract_message', [])
        model = session.get('mat_model')
        predictions = session.get('mat_prediction', [])
        performance_metrics = session.get('download_performance', {})
        
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('title_style', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=18, spaceAfter=12, textColor=black)
        heading_style = ParagraphStyle('heading_style', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, spaceAfter=6, textColor=black)
        body_style = ParagraphStyle('body_style', parent=styles['BodyText'], fontName='Helvetica', fontSize=12, leading=15, textColor=black)

        # Title
        elements.append(Paragraph("EEG Classification Report for MAT Files", title_style))

        # Subtitle and Introduction
        intro_text = f"<b>Report generated by:</b> EEG Hub Analysis System<br/>"\
                     f"<b>Classification model used:</b> <font color='blue'>{model}</font><br/>"\
                     "This document provides a detailed overview of the EEG classification process, "\
                     "including data preprocessing, feature extraction, predictions, and performance metrics."
        elements.append(Paragraph(intro_text, body_style))
        elements.append(Spacer(1, 12))

        # Preprocessing Steps
        elements.append(Paragraph("<b>Preprocessing Steps:</b>", heading_style))
        for step in preprocess_steps:
            elements.append(Paragraph(step, body_style))

        # Feature Extraction Steps
        elements.append(Paragraph("<b>Feature Extraction Steps:</b>", heading_style))
        for message in extract_messages:
            elements.append(Paragraph(message, body_style))

        # Predictions
        elements.append(Paragraph("<b>Predictions:</b>", heading_style))
        for prediction in predictions:
            if isinstance(prediction, dict):
                prediction_text = '<br/>'.join([f"{k}: {v}" for k, v in prediction.items()])
            else:
                prediction_text = str(prediction)
            elements.append(Paragraph(prediction_text, body_style))

        # Performance Metrics
        elements.append(Paragraph("<b>Performance Metrics:</b>", heading_style))
        for key, value in performance_metrics.items():
            metric_text = f"<b>{key}:</b> {value}"
            elements.append(Paragraph(metric_text, body_style))

        # Build PDF
        doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)

        # Move to the beginning of the BytesIO buffer
        pdf_buffer.seek(0)
        return send_file(pdf_buffer, as_attachment=True, download_name="EEG_Classification_Report.pdf", mimetype='application/pdf')

    except Exception as e:
        print(traceback.format_exc())  # This will print the full traceback to the console
        return jsonify({'error': str(e)}), 500    