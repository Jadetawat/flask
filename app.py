from flask import Flask, render_template, url_for, request, redirect,send_from_directory
from werkzeug.utils import secure_filename
import os
from script import pdf2img, png2jpg, crop_info, OCRextract, invoice_extract, order_extract, receipt_extract, t_extract, json_receipt, json_invoice, json_sale_order
from PIL import Image
from huggingface_hub import hf_hub_download


import numpy as np
import pandas as pd
import json
import csv
import shutil

app = Flask(__name__)


ALLOWED_EXTENSIONS = set(['pdf','png','jpg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET',"POST"])

def upload():
    
    return render_template('index.html')
   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
