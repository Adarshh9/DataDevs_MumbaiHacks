from flask import Flask, jsonify, request,send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)


@app.route('/')
def Landing():
    return("Landing Page")



UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    # Add any file extension checks here if needed
    return '.' in filename

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("File Uploaded Sucessfully")
        return 'File uploaded successfully'
    else:
        return 'Invalid file'

@app.route('/files/<path:filename>', methods=['GET'])
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/store_option/<option>', methods=['POST'])
def store_option(option):
    global selected_option
    selected_option = option
    print(option)
    return 'Option stored successfully!'

@app.route('/get_option', methods=['GET'])
def get_option():
    global selected_option
    return selected_option or 'No option selected'

stored_link = None

@app.route('/store_link/<path:link>', methods=['POST'])
def store_link(link):
    global stored_link
    stored_link = link
    return 'Link stored successfully!'

@app.route('/get_link', methods=['GET'])
def get_link():
    global stored_link
    return stored_link or 'No link stored'



if(__name__ == "__main__"):
    app.run(port=8000, debug=True)