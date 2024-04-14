import webbrowser
from threading import Timer
from flask import Flask, render_template, session,flash
from frontend.views import views
from components.file_reader import bp_file_reader
app = Flask(__name__)

app.register_blueprint(bp_file_reader)
app.secret_key = 'csv_messages'

@app.route('/')
def home():
     return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/plots')
def plots():
     return render_template('plots.html')

@app.route('/features')
def features():
     return render_template('features.html')

@app.route('/results')
def results():
     return render_template('results.html')

@app.route('/csv')
def csv_files():
    # Retrieve messages from session
    messages = session.get('csv_messages', [])
    return render_template('csv_files.html', messages=messages)

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()  # Wait 1 second before opening the browser
    app.run(debug=True, use_reloader=False)