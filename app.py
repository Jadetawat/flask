from flask import Flask, render_template, url_for, request,redirect,send_from_directory
from werkzeug.utils import secure_filename
import os
from script import pdf2img, png2jpg, information_extract
from PIL import Image
import pandas as pd
import secrets
import tempfile

app = Flask(__name__)
key = secrets.token_hex(16)
app.secret_key = key


ALLOWED_EXTENSIONS = set(['pdf','png','jpg','jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Modify the upload function to create a private directory for each user
user_dir=tempfile.mkdtemp()
process_dir=tempfile.mkdtemp()
output_dir=tempfile.mkdtemp()
@app.route('/', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Generate unique directory names for each user session
            save_location = os.path.join( user_dir, filename)
            input_process = os.path.join( process_dir, filename)
            file.save(save_location)
            cropped_table_path=os.path.join( process_dir, "cropped_table.jpg")
            removed_table_path=os.path.join( process_dir, "removed_table.jpg")
            csv_output=os.path.join( output_dir, "output.csv")
            json_output=os.path.join( output_dir, "output.json")
            Json_Table_Path=os.path.join( process_dir, "table.json")

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

            return redirect(url_for('download'))

    return render_template('index.html')


# Modify the download function to display only files from the user's directory
@app.route('/download')
def download():

    files = os.listdir(output_dir)
    df = pd.DataFrame()
    try:
        df = pd.read_csv(os.path.join(output_dir, "output.csv"), encoding="utf-8")
    except Exception as e:
        print(e)
    return render_template('download.html', files=files, tables=[df.to_html(index=False, classes='data', header="true")])

@app.route('/download/<filename>')
def download_file(filename):

        return send_from_directory(directory=output_dir, filename=filename, as_attachment=True)

    

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5000)
