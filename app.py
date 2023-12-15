from flask import Flask, jsonify, request
from flask_pymongo import PyMongo

app = Flask(__name__)


@app.route('/')
def Landing():
    return("Landing Page")

# Adding blog
@app.route('/dashboard/user_domain/<domain_name>' )
def dash_userdomain(domain_name):
    return("Domain Name" , domain_name)



if(__name__ == "__main__"):
    app.run(port=8000, debug=True)