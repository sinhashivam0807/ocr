from flask import Flask, jsonify,request

import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
import base64
from dotenv import load_dotenv
load_dotenv()

API_KEY=os.getenv("API_KEY")
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'

class ImageConstantROI():
    class CCCD(object):
        ROIS = {
            "birth_date": [(426,482,252, 45)],
            "sex": [(742, 489, 100, 52)],
            "issue_date": [(425, 594, 244, 40)],
            "expiry_date": [(673, 594, 290, 40)],
            
        }
#importing base image
baseImg = cv2.imread('base.png')
baseH, baseW, baseC = baseImg.shape

#Cropped Image
def cropImageRoi(image, roi):
    roi_cropped = image[
        int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
    ]
    return roi_cropped

def preprocessing_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.multiply(gray, 1.5)
    blured1 = cv2.medianBlur(gray,3)
    blured2 = cv2.medianBlur(gray,51)
    divided = np.ma.divide(blured1, blured2).data
    normed = np.uint8(255*divided/divided.max())
    th, threshed = cv2.threshold(normed, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return threshed

#test image

def image2(img):
    #Init orb, keypoints detection on base Image
    orb = cv2.ORB_create(1000)
    kp, des = orb.detectAndCompute(baseImg, None)
    imgKp = cv2.drawKeypoints(baseImg,kp, None)
    PER_MATCH = 0.25

#Detect keypoi25nt on img2
    kp1, des1 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(bf.match(des1, des))
    matches.sort(key=lambda x: x.distance)
    best_matches = matches[:int(len(matches)*PER_MATCH)]

#Init source points and destination points for findHomography function.
    srcPoints = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1,1,2)
    dstPoints = np.float32([kp[m.trainIdx].pt for m in best_matches]).reshape(-1,1,2)
    matrix_relationship, _ = cv2.findHomography(srcPoints, dstPoints,cv2.RANSAC,6.0)
    img_final = cv2.warpPerspective(img, matrix_relationship, (baseW, baseH))
    return img_final

#rest data
MODEL_CONFIG =r'-l eng --oem 3 --psm 6'
def extractDataFromIdCard(img):
    output={}
    for key, roi in ImageConstantROI.CCCD.ROIS.items():
        data = ''
        for r in roi:
            crop_img = cropImageRoi(img, r)
            crop_img = preprocessing_image(crop_img)
            #Extract data from image using pytesseract
            data+= pytesseract.image_to_string(crop_img, config = MODEL_CONFIG).replace(" ", "") + ' '
            output_data=re.sub('[QWERTYUIOPASDqwertyuiopasdghjklzxcvbnGHJKLZXCVBN~!@#$%^&*((_+={|\:?.))]', '',data)
            print(output_data)
        output[key]=output_data.strip()
    return output

def ocr_output():
    img2 = cv2.imread('testimage.png')
    json_data={}
    img_final=image2(img2)
    json_data=extractDataFromIdCard(img_final)
    #signature
    roi_sign= (55, 570,364,128)
    signature=cropImageRoi(img_final, roi_sign)
    cv2.imwrite("sign.png", signature);
    with open("sign.png", "rb") as image_file:
        image64 = base64.b64encode(image_file.read()).decode('utf-8')

#name and health id
    custom_config = '-l eng --oem 3 --psm 6'
    roi_name=(412,194,700,258)
    name_img=cropImageRoi(img_final, roi_name)
    name_preprocess=preprocessing_image(name_img)
    text_name=pytesseract.image_to_string(name_preprocess,lang='eng', config=custom_config)
    
    roi_sign=(412,176,551,268)
    sign_img=cropImageRoi(img_final, roi_sign)
    sign_preprocess=preprocessing_image(sign_img)
    text_number=pytesseract.image_to_string(name_preprocess,lang='eng', config=custom_config).replace(" ", "")

    name_data= re.findall(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', text_name)
    health_data= re.findall('\d{4}-\d{3}-\d{3}-\w{2}',text_number)
    if(len(name_data)==0):
        json_data["name"]=""
    else:
        name_output=re.sub('[\n\t.!@#$%^&*()_+={}[]\|::<>?/123456789]', '', name_data[0])
        json_data["name"]=name_output
    if(len(health_data)==0):
        json_data["health_id"]=""
    else:
        json_data["health_id"]=health_data[0]
        json_data["signature"]=image64
        json_obj=jsonify(json_data)
        return json_obj

app= Flask(__name__)
CORS(app)

@app.route('/autofillform', methods = ['POST'])
def ReturnJSON():
    if request.method == "POST":
        img2get=(request.json.get('image'))
        with open("testimage.png", "wb") as fh:
            fh.write(base64.b64decode(img2get))
        if(request.headers.get('API_KEY')==API_KEY):
            if(ocr_output()==None):
                return("Please enter image in correct orientation")
            else:
                return (ocr_output())
        else:
            return("Authorization Failed !!")

if __name__ == '__main__':
    app.run(debug=False,ssl_context="adhoc")

