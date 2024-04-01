from flask import Flask
from views import views

app= Flask(__name__)
app.register_blueprint(views, url_prefix="/views")



if __name__ == '__main__':
    app.run(debug=True ,port=8000) #default port 5000 , debug is true means if u changed anything in the file it automatically updates iit on the website
