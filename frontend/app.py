import webbrowser
from threading import Timer
from flask import Flask
from frontend.views import views
app = Flask(__name__)

app.register_blueprint(views, url_prefix='/')

@app.route('/')
def hello():
    return 'Hello, World!'

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()  # Wait 1 second before opening the browser
    app.run(debug=True, use_reloader=False)