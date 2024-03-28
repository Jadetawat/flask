from flask import Flask,render_template,url_for,request,redirect,send_from_directory
from werkzeug.utils import secure_filename
import os
from script import pdf2img, png2jpg, information_extract
from PIL import Image
import pandas as pd
import tempfile

app = Flask(__name__)
user_dir=tempfile.mkdtemp()
process_dir=tempfile.mkdtemp()
output_dir=tempfile.mkdtemp()
ALLOWED_EXTENSIONS = set(['pdf','png','jpg','jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_location = os.path.join( user_dir, filename)
            input_process = os.path.join( process_dir, filename)
            file.save(save_location)
            cropped_table_path=os.path.join( process_dir, "cropped_table.jpg")
            removed_table_path=os.path.join( process_dir, "removed_table.jpg")
            csv_output=os.path.join( output_dir, "output.csv")
            json_output=os.path.join( output_dir, "output.json")
            Json_Table_Path=os.path.join( process_dir, "table.json")
            if filename.lower().endswith(('.pdf')):
                filename = filename.split('.')[0] + '.jpg'
                input_process = os.path.join(process_dir, filename)
                pdf2img(save_location, input_process)
            elif filename.lower().endswith(('.png')):
                filename = filename.split('.')[0] + '.jpg'
                input_process = os.path.join(process_dir, filename)
                png2jpg(save_location, input_process)
            else:
                image=Image.open(save_location,'r')
                image.save(input_process, 'JPEG')
            im = Image.open(input_process,'r')
            try:
                information_extract(request.form['format']
                                    ,im,cropped_table_path
                                    ,removed_table_path
                                    ,csv_output,json_output
                                    ,Json_Table_Path)
            except Exception as e: print(e)
            return redirect(url_for('download'))
    return render_template('index.html')

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
