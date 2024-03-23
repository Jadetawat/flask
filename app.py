from flask import Flask, render_template, url_for, request, redirect,send_from_directory
from werkzeug.utils import secure_filename
import os
from script import pdf2img, png2jpg, information_extract
from PIL import Image
import pandas as pd
import shutil

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['pdf','png','jpg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET',"POST"])

def upload():
    try:
        if os.path.exists("./process"):
            shutil.rmtree("./process")
        else:
            print("process folder does not exist")

        if os.path.exists("./input"):
            shutil.rmtree("./input")
        else:
            print("input folder does not exist")

        if os.path.exists("./output"):
            shutil.rmtree("./output")
        else:
            print("output folder does not exist")

        os.mkdir("./process")
        os.mkdir("./input") 
        os.mkdir("./output")
    except Exception as e: print(e)    
    if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_location = os.path.join('input', filename)
                input_process  = os.path.join('process', filename) 
                file.save(save_location)

                if filename.lower().endswith(('.pdf')):
                    pdf2img(save_location,filename.split('.')[0])
                    filename=filename.split('.')[0]+'.jpg'
                    input_process  = os.path.join('process', filename) 
                elif filename.lower().endswith(('.png')):
                    png2jpg(save_location,filename.split('.')[0])
                    filename=filename.split('.')[0]+'.jpg'
                    input_process  = os.path.join('process', filename) 
                else:
                    with Image.open(save_location) as image:
                        image.save(input_process , 'JPEG')
                        image.close()
     
                with Image.open(input_process) as im:
                    information_extract(request.form['format'],im)

                return redirect(url_for('download'))

    return render_template('index.html')
   
   
@app.route('/download')
def download():
    df = pd.DataFrame()
    try:
        df = pd.read_csv("./output/output.csv",encoding="utf-8")
    except Exception as e: print(e)
    return render_template('download.html', files=os.listdir('output'),tables=[df.to_html(index = False,classes='data', header="true")])

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
