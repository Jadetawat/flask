from flask import Flask, render_template, url_for, request,session,redirect,send_from_directory
from werkzeug.utils import secure_filename
import os
from script import pdf2img, png2jpg, information_extract
from PIL import Image
import pandas as pd
import shutil
import secrets
import time

app = Flask(__name__)
key = secrets.token_hex(16)
app.secret_key = key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 180

ALLOWED_EXTENSIONS = set(['pdf','png','jpg','jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_token():
    return secrets.token_hex(8)

# Modify the upload function to create a private directory for each user

@app.route('/', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Generate unique directory names for each user session
            user_dir=generate_token()
            process_dir=generate_token()
            output_dir=generate_token()
            os.makedirs(os.path.join('input', user_dir))
            os.makedirs(os.path.join('process', process_dir))
            os.makedirs(os.path.join('output', output_dir))
            save_location = os.path.join('input', user_dir, filename)
            input_process = os.path.join('process', process_dir, filename)
            file.save(save_location)
            cropped_table_path=f"./process/{process_dir}/cropped_table.jpg"
            removed_table_path=f"./process/{process_dir}/removed_table.jpg"
            csv_output=f"./output/{output_dir}/output.csv"
            json_output=f"./output/{output_dir}/output.json"
            Json_Table_Path=f"./process/{process_dir}/table.json"

            if filename.lower().endswith(('.pdf')):
                pdf2img(save_location, filename.split('.')[0])
                filename = filename.split('.')[0] + '.jpg'
                input_process = os.path.join('process', process_dir, filename)
            elif filename.lower().endswith(('.png')):
                png2jpg(save_location, filename.split('.')[0])
                filename = filename.split('.')[0] + '.jpg'
                input_process = os.path.join('process', process_dir, filename)
            else:
                with Image.open(save_location) as image:
                    image.save(input_process, 'JPEG')

            with Image.open(input_process) as im:
                information_extract(request.form['format'],im,cropped_table_path,removed_table_path,csv_output,json_output,Json_Table_Path)
                
 
            
            # Store the user_dir, process_dir, and output_dir in the session
            session['user_dir'] = user_dir
            session['process_dir'] = process_dir
            session['output_dir'] = output_dir

            return redirect(url_for('download'))

    return render_template('index.html')


# Modify the download function to display only files from the user's directory
@app.route('/download')
def download():
    user_dir = session.get('user_dir')
    process_dir = session.get('process_dir')
    output_dir = session.get('output_dir')

    if not (user_dir and process_dir and output_dir):
        return redirect(url_for('upload'))  # Redirect to upload if user_dir, process_dir, or output_dir is not set

    user_output_dir = os.path.join('output', output_dir)
    files = os.listdir(user_output_dir)
    df = pd.DataFrame()
    try:
        df = pd.read_csv(os.path.join(user_output_dir, "output.csv"), encoding="utf-8")
    except Exception as e:
        print(e)
    return render_template('download.html', files=files, tables=[df.to_html(index=False, classes='data', header="true")])

@app.route('/download/<filename>')
def download_file(filename):
    user_output_dir = session.get('output_dir')
    if user_output_dir:
        return send_from_directory(directory=os.path.join('output', user_output_dir), filename=filename, as_attachment=True)
    else:
        return render_template('index.html')
    
@app.errorhandler(Exception)
def handle_exception(e):
    # Perform cleanup operations here
    user_dir = session.pop('user_dir', None)
    process_dir = session.pop('process_dir', None)
    output_dir = session.pop('output_dir', None)
    if user_dir:
        shutil.rmtree(os.path.join('input', user_dir))

    if process_dir:
        shutil.rmtree(os.path.join('process', process_dir))

    if output_dir:
        shutil.rmtree(os.path.join('output', output_dir))

    # Log the exception or perform other error handling tasks
    # You can return a custom error page or response here if needed
    return 'An error occurred', 500

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5000)
