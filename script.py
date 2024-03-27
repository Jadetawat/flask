from PIL import Image,ImageDraw
from transformers import TableTransformerForObjectDetection,AutoModelForObjectDetection,DetrImageProcessor
import torch
import numpy as np
import pandas as pd
import json
import easyocr
import fitz  # PyMuPDF
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
reader = easyocr.Reader(['th','en'])
table_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection",revision="no_timm")
device = "cuda" if torch.cuda.is_available() else "cpu"
table_model.to(device)
feature_extractor = DetrImageProcessor()
Tstructure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")

def pdf2img(pdf_path, input_process, dpi=300):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(input_process, dpi=(dpi, dpi))
    pdf_document.close()

def png2jpg(image_path, input_process):  
  with Image.open(image_path) as img:
    img.convert('RGB').save(f"{input_process}", 'JPEG')
     
def blank(im,bbox,removed_table_path):
  fill_white = ImageDraw.Draw(im)
  start_point = (bbox[0],bbox[1])
  end_point = (bbox[2],bbox[3])
  fill_white.rectangle((start_point,end_point), fill ="white")
  im.save(removed_table_path)

def tableDetect(image,cropped_table_path,removed_table_path,padding=0):
    encoding = feature_extractor(image, return_tensors="pt")
    encoding.keys()
    width, height = image.size
    with torch.no_grad():
        outputs = table_model(**encoding)
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.8, target_sizes=[(height, width)])[0]
    bbox=[results['boxes'][0][0].tolist()-padding, results['boxes'][0][1].tolist()-padding, results['boxes'][0][2].tolist()+padding, results['boxes'][0][3].tolist()+padding]
    cropped_table=image.crop(bbox)
    blank(image,bbox,removed_table_path)
    cropped_table.save(cropped_table_path)

def compute_boxes(cropped_table,threshold):
    encoding = feature_extractor(cropped_table, return_tensors="pt")
    encoding.keys()
    with torch.no_grad():
        outputs = Tstructure_model(**encoding)
    target_sizes = [cropped_table.size[::-1]]
    results = feature_extractor.post_process_object_detection(outputs, threshold, target_sizes=target_sizes)[0]
    boxes = results['boxes'].tolist()
    labels = results['labels'].tolist()
    return boxes,labels

def tableRecognize(img_path,threshold=0.8):
    cropped_table=Image.open(img_path)
    boxes,labels = compute_boxes(cropped_table,threshold)
    cell_locations = []
    for box_row, label_row in zip(boxes, labels):
        if label_row == 2:
            for box_col, label_col in zip(boxes, labels):
                if label_col == 1:
                    cell_box = (box_col[0], box_row[1], box_col[2], box_row[3])
                    cell_locations.append(cell_box)
    cell_locations.sort(key=lambda x: (x[1], x[0]))

    empty_row=0
    num_columns = 0
    box_old = cell_locations[0]
    for box in cell_locations[1:]:
        x1, y1, x2, y2 = box
        x1_old, y1_old, x2_old, y2_old = box_old
        num_columns += 1
        if y1 > y1_old:
            break
        box_old = box
        
    df = pd.DataFrame(columns=range(1,num_columns+1)).add_prefix("column")
    row = []
    for box in cell_locations[:]:
        x1, y1, x2, y2 = box
        cell_image = np.array(cropped_table.crop((x1, y1, x2, y2)))
        cell_text = reader.readtext(np.array(cell_image),paragraph=False, detail=0)
        if len(cell_text) == 0:
          cell_text=np.nan
        else:
          cell_text = ' '.join([str(elem) for elem in cell_text])
        row.append(cell_text)
        if len(row) == num_columns:
            if all(pd.isnull(row)):
              empty_row+=1
              if empty_row==2:
                return df
            else: empty_row=0
            df.loc[len(df)] = row
            row = []
    cropped_table.close()
    return df

def behind(lower,text):
  for i in range(0,len(text),2):
    if text[i].lower() == lower:
      data=text[i+1].strip(" ")
      return data

def crop2OCR(img_path,pos):
    img = Image.open(img_path).convert("RGB")
    cropped_img=img.crop(pos)
    arr = np.array(cropped_img)
    text = reader.readtext(np.array(arr),paragraph=False, detail=0)
    img.close()
    return text

def json_table(json_output,Json_Table_Path):
  with open(Json_Table_Path, encoding='utf-8') as json_file:
    add_value = json.load(json_file)
    json_file.close()
  with open( json_output, encoding='utf-8') as json_file:
    json_decoded = json.load(json_file)
    json_decoded[0]['table'] = add_value
    json_file.close()
  with open(json_output, 'w', encoding='utf-8') as json_file:
    json.dump(json_decoded, json_file,ensure_ascii=False)
    json_file.close()

def information_extract(format,im,cropped_table_path,removed_table_path,csv_output,json_output,Json_Table_Path):
  width, height = im.size
  if format=='001':
    table=[120*width/2616, 1127*height/3385, 2552*width/2616, 1772*height/3385]
    cropped_table=im.crop(table)
    cropped_table.save(cropped_table_path)
    blank(im,table,removed_table_path)

    date = [width*333/2616,height*922/3385,width*570/2616,height*960/3385]
    date_crop=crop2OCR(removed_table_path,date)
    total=[width*2300/2616,height*1800/3385,width*2545/2616,height*2170/3385]
    total_crop=crop2OCR(removed_table_path,total)
    df=tableRecognize(cropped_table_path,0.8)
    df=df.dropna(how='all')   
    try:
      df.columns = ['ลำดับ', 'รหัสสินค้า', 'รายการสินค้า', 'รายละเอียด/สี', 'XS', 'S', 'M', 'L', 'XL', 'รวมจำนวน', 'ราคา', 'รวมราคา']
      df.to_json(Json_Table_Path,force_ascii=False, orient ='records')
    except:
      print("columns must be unique")
    
    df['Due_Date']=date_crop[0]
    df['ราคาสินค้าก่อนหักภาษี']=total_crop[0]
    df['ภาษี ณ ที่จ่าย3%']=total_crop[1]
    df['ค่ามัดจำงาน']=total_crop[2]
    df['ค่ามัดจำงาน']=total_crop[3]
    df['รวมสุทธิ']=total_crop[4]
    df['ค่ามัดจำงานผลิต70%']=total_crop[5]
    df['คงเหลือ']=total_crop[6]
    df.to_csv(csv_output, index=False,encoding="utf-8")
    information = np.array([[date_crop[0] ,total_crop[0] ,total_crop[1] ,total_crop[2] ,total_crop[3] ,total_crop[4] ,total_crop[5] ,total_crop[6] ]])
    df1 = pd.DataFrame(information, columns = ['วันนัดส่ง','ราคา สินค้าก่อนหักภาษี ณ ที่จ่าย 3%', 'ภาษี ณ ฺที่จ่าย 3%', 'รวมราคาคงเหลือ', 'ค่ามัดจำงาน', 'รวมสุทธิ', 'ค่ามัดจำงานผลิต 70 %', 'คงเหลือ'])
    df1['table']=np.nan
    df1.to_json( json_output,force_ascii=False, orient ='records')
    try:
      json_table(json_output,Json_Table_Path)
    except:
      print("table.json is not found")
    
  elif format=='002':  
    tableDetect(im,cropped_table_path,removed_table_path,30)
    date = [width*1627/2481,height*487/3508,width*1893/2481,height*535/3508]
    date_crop=crop2OCR(removed_table_path,date)
    total=[width*1670/2481,height*1000/3508,width*2370/2481,height*2880/3508]
    total_crop=crop2OCR(removed_table_path,total)
    df=tableRecognize(cropped_table_path,0.8)
    df = df.dropna(how="all")
    df.columns = df.iloc[0]
    df = df[1:]
    try:
      df.columns = ['#', 'Description', 'Quantity', 'Unit Price', 'Total']
      df.to_json(Json_Table_Path,force_ascii=False, orient ='records')
    except:
      print("columns must be unique")

    df['Due_Date']=date_crop[0]
    df['invoice_Total']=behind('total',total_crop)
    df['invoice_Discount']=behind('discount',total_crop)
    df['Total_after_discount']=behind('total after discount',total_crop)
    df['Grand_Total']=behind('grand total',total_crop)
    df.to_csv(csv_output, index=False,encoding="utf-8")
    
    information = np.array([[date_crop[0] ,df['invoice_Total'].tolist()[0] ,df['invoice_Discount'].tolist()[0] ,df['Total_after_discount'].tolist()[0] ,df['Grand_Total'].tolist()[0]]] )
    df1 = pd.DataFrame(information, columns = ['Due_Date','invoice_Total', 'invoice_Discount', 'Total_after_discount', 'Grand_Total'])
    df1['table']=np.nan
    df1.to_json( json_output,force_ascii=False, orient ='records')
    try:
      json_table(json_output,Json_Table_Path)
    except:
      print("table.json is not found")

  elif format=='003':
    table=[width*101/5296,height*2397/7488,width*5161/5296,height*5053/7488]
    cropped_table=im.crop(table)
    cropped_table.save(cropped_table_path)
    df=tableRecognize(cropped_table_path,0.8)
    df = df.dropna(how='all')
    df.columns = df.iloc[0]
    df = df[1:] 
    try:
      df.columns = ['No.', 'รหัสสินค้า/รายละเอียด', 'จำนวน', 'หน่วยละ', 'จำนวนเงิน']
      df.to_json(Json_Table_Path,force_ascii=False, orient ='records')
    except:
      print("columns must be unique")
     
    blank(im,table,removed_table_path)
    total=[width*4477/5296,height*6065/7488,width*5105/5296,height*6205/7488]
    total_crop=crop2OCR(removed_table_path,total)
    date = [width*4613/5296,height*1407/7488,width*5129/5296,height*1557/7488]
    date_crop=crop2OCR(removed_table_path,date)
    df['date']=date_crop[0]
    df['total']=total_crop[0]

    df.to_csv(csv_output, index=False,encoding="utf-8")
    information = np.array([[date_crop[0] ,total_crop[0]]])
    df1 = pd.DataFrame(information, columns = ['date','total'])
    df1['table']=np.nan
    df1.to_json( json_output,force_ascii=False, orient ='records')
    try:
      json_table(json_output,Json_Table_Path)
    except:
      print("table.json is not found")
    return df

  else:
    tableDetect(im,cropped_table_path,removed_table_path,30)
    df=tableRecognize(cropped_table_path,0.8)
    df.dropna(how="all")
    df.columns = df.iloc[0]
    df = df[1:]       
    df.to_csv(csv_output, index=False,encoding="utf-8")    
    try:
      df.to_json(json_output,force_ascii=False, orient ='records')
    except:
      print("columns must be unique")


  


