import requests
import json
import cv2
import os
from flask import Flask, request, render_template, url_for, session, jsonify, make_response, redirect
import pytesseract
import matplotlib.pyplot as plt
import shutil
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['GET'])
def selectPageMode():
    return render_template('select_mode.html')

@app.route('/index',methods=['GET'])
def index():
    filename = ''
    config_filename = ''
    rectnames = ''
    rectangles = ''
    scale_img= ''
    if 'filename' in session:
        filename = session['filename']
    # if 'config_filename' in session:
    #     config_filename = session['config_filename']
    # if 'rectnames' in session:
    #     rectnames = session['rectnames']
    # if 'rectangles' in session:
    #     rectangles = session['rectangles']
    # if 'scale_img' in session:
    #     scale_img = session['scale_img']
    return render_template('view.html', filename = filename, config_filename = config_filename, rectnames = rectnames, rectangles = rectangles, scale_img=scale_img)

@app.route('/uploadImage', methods=['POST'])
def viewImage():
    #print(request)
    #session = None
    image = request.files['file']
    if image.filename == '':
        return index()
    print(image.filename)
    filename = 'static/uploaded/' + image.filename.replace(image.filename[-4:],'.png')
    image.save(filename)
    print('Write file success!',filename)
    session['filename'] = filename
    # if 'config_filename' in session:
    #     config_filename = session['config_filename']
    # else:
    #     config_filename = ''
    return render_template('view.html', filename = filename, config_filename = '', rectnames = '', rectangles = '', scale_img= '')

from eastDetect import detect
@app.route('/autoDetect', methods=['POST'])
def detectRects():
    rq = request.get_json()
    #rectangle = rq['cur_rectangle']
    scale_img = rq['scale_img']*0.6

    if 'filename' not in session:
        return index()
    filename = session['filename']
    print('Detecting...')
    rectangles, rectnames = detect(filename, scale_img)
    print('Detect done')
    #rectangles = rectangles
    print(rectangles, len(rectangles))
    print(rectnames, len(rectnames))
    resp = jsonify(success=True,rectangles=rectangles, rectnames=rectnames, scale_img=scale_img)
    return resp
    #return render_template('view.html', filename = filename, config_filename = '', rectnames = rectnames, rectangles = rectangles, scale_img= scale_img)

@app.route('/uploadConfig', methods=['POST'])
def viewConfig():
    if 'filename' not in session:
        return index()
    a = {}
    try:
        config = request.files['file'].read()
        config = json.loads(config)
        a = json.dumps(config)
    except:
        return index()
    print(config)
    if 'config_filename' not in config:
        return index()
    print(config['rectangles'])
    #config_filename = 'static/uploaded/' + config['config_filename'] + '.json'
    config_filename = config['config_filename']
    rectnames = []
    rectangles = []
    scale_img = float(config['scale_img'])
    for rect in config['rectangles']:
        rectnames.append(rect['rectname'])
        rectangles.append(rect['rectangle'])
    print(rectnames)
    print(rectangles)
    print(scale_img)
    session['config_filename'] = config_filename
    # session['rectnames'] = rectnames
    # session['rectangles'] = rectangles
    # session['scale_img'] = scale_img
    return render_template('view.html', filename = session['filename'], config_filename = session['config_filename'], rectnames = rectnames, rectangles = rectangles, scale_img=scale_img)

@app.route('/save_config', methods=['POST'])
def saveConfig():
    rq = request.get_json()
    save_filename = rq['config_filename']
    rectangles = rq['rectangles']
    scale_img = rq['scale_img']
    #print(rectangles)
    save_filename = 'static/mau/'+save_filename+'.json'
    if os.path.exists(save_filename):
        print('file is exists:',save_filename)
        resp = jsonify(success=False, error='File:'+save_filename+' is exists, try orther name!')
    else:
        shutil.copy(session['filename'], save_filename.replace('.json','.png'))
        with open(save_filename,'w') as f:
            json.dump(rq,f)
        print('write json success:',save_filename)
        resp = jsonify(success=True)
    return resp

@app.route('/recognize_one_rectangle',methods=['POST'])
def recognizeCurrectangle():
    rq = request.get_json()
    rectangle = rq['cur_rectangle']
    scale_img = rq['scale_img']*0.6
    x = rectangle['x']/scale_img
    y = rectangle['y']/scale_img
    w = rectangle['width']/scale_img
    h = rectangle['height']/scale_img
    print('current rect:',rectangle)
    filename = session['filename']
    #print(filename)
    img = cv2.imread(filename)
    print(img.shape)
    crop_img = img[int(y):int(y+h), int(x):int(x+w)]
    #plt.imshow(crop_img[:,:,0])
    #plt.show()
    print(crop_img.shape)
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    #crop = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    #crop = cv2.GaussianBlur(crop, (1, 1), 0)
    #_, crop = cv2.threshold(crop, 120 ,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #plt.imshow(crop)
    #plt.show()
    text = pytesseract.image_to_string(crop_img, config='-l vie --oem 1')
    print('text_cur_rect:',text)
    resp = jsonify(success=True,data=text)
    return resp

@app.route('/recognize_all_rectangles',methods=['POST'])
def recognizeAllrectangle():
    rq = request.get_json()
    rectangles = rq['rectangles']
    scale_img = rq['scale_img']*0.6
    print('rectangles:',rectangles, len(rectangles))
    filename = session['filename']
    img = cv2.imread(filename)
    print(img.shape)
    texts = []
    for rectangle in rectangles:
        x = rectangle['x']/scale_img
        y = rectangle['y']/scale_img
        w = rectangle['width']/scale_img
        h = rectangle['height']/scale_img
        #print(filename)
        print(rectangle['x'],rectangle['y'],rectangle['width'],rectangle['height'])
        crop_img = img[int(y):int(y+h), int(x):int(x+w)]
        #plt.imshow(crop_img[:,:,0])
        #plt.show()
        print(crop_img.shape)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_img = unsharp_mask(crop_img)
        plt.imshow(crop_img[:,:,0])
        plt.show()
        #crop = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        #crop = cv2.GaussianBlur(crop, (1, 1), 0)
        #_, crop = cv2.threshold(crop, 120 ,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        #plt.imshow(crop)
        #plt.show()
        text = pytesseract.image_to_string(crop_img, config='-l vie --oem 1')
        print('text_cur_rect:',text)
        texts.append(text)
    print(texts)
    resp = jsonify(success=True,data=texts)
    return resp
    
@app.route('/index2')
def index2():
    filename = ''
    config_filename = ''
    rectnames = ''
    rectangles = ''
    scale_img= ''
    list_config_files = []
    if 'filename2' in session:
        filename = session['filename2']
    # if 'config_filename2' in session:
    #     config_filename = session['config_filename2']
    # if 'rectnames2' in session:
    #     rectnames = session['rectnames2']
    # if 'rectangles2' in session:
    #     rectangles = session['rectangles2']
    # if 'scale_img2' in session:
    #     scale_img = session['scale_img2']
    for f in os.listdir('static/mau'):
        print(f.split('.')[-1])
        if f.split('.')[-1] == 'json':
            list_config_files.append(f.split('.')[0])
    print(list_config_files)
    session['list_config_files'] = list_config_files
    return render_template('view2.html', filename = filename, list_config_files = list_config_files, config_filename = config_filename, rectnames = rectnames, rectangles = rectangles, scale_img=scale_img)

@app.route('/uploadImage2', methods=['POST'])
def uploadImage2():
    image = request.files['file']
    if image.filename == '':
        return index()
    print(image.filename)
    filename = 'static/uploaded/' + image.filename.replace(image.filename[-4:],'.png')
    image.save(filename)
    #print('Write file success! '+filename)
    session['filename2'] = filename
    return render_template('view2.html', filename = filename, list_config_files = session['list_config_files'], config_filename = '', rectnames = '', rectangles = '', scale_img= '')

@app.route('/demo_<string:demo_img>', methods=["GET"])
def demo(demo_img):
    demo_img = demo_img.replace('_','/')
    print(demo_img)
    shutil.copy(demo_img,demo_img.replace('demo','uploaded'))
    session['filename2'] = demo_img.replace('demo','uploaded')
    return index2()

@app.route('/uploadConfig2', methods=['POST'])
def viewConfig2_byupload():
    if 'filename2' not in session:
        print("filename not in session")
        return index2()
    a = {}
    try:
        config = request.files['file2'].read()
        print(config)
        config = json.loads(config)
        a = json.dumps(config)
    except:
        print('read config request error')
        return index2()
    print(config)
    if 'config_filename' not in config:
        print('config_filename not in config')
        return index2()
    print(config['rectangles'])
    config_filename = config['config_filename']
    rectnames = []
    rectangles = []
    scale_img = float(config['scale_img'])
    for rect in config['rectangles']:
        rectnames.append(rect['rectname'])
        rectangles.append(rect['rectangle'])
    print(rectnames)
    print(rectangles)
    print(scale_img)
    session['config_filename2'] = config_filename
    print(rectnames)
    return render_template('view2.html', filename = session['filename2'], list_config_files = session['list_config_files'], config_filename = session['config_filename2'], rectnames = rectnames, rectangles = rectangles, scale_img=scale_img)

@app.route('/<string:config_filename>',methods=["GET"])
def viewConfig_byselect(config_filename):
    session['config_filename2'] = config_filename
    config = []
    rectnames = []
    rectangles = []
    scale_img = 0.0
    with open('static/mau/' + config_filename + '.json') as f:
        config = json.load(f)
        scale_img = float(config['scale_img'])
        for rect in config['rectangles']:
            rectnames.append(rect['rectname'])
            rectangles.append(rect['rectangle'])
    return render_template('view2.html', filename = session['filename2'], list_config_files = session['list_config_files'], config_filename = session['config_filename2'], rectnames = rectnames, rectangles = rectangles, scale_img=scale_img)

@app.route('/selectconfigandalign_<string:config_filename>')
def selectConfigAndAlign(config_filename):
    if 'filename2' not in session:
        return index2()
    session['config_filename2'] = config_filename
    config = []
    rectnames = []
    rectangles = []
    scale_img = 0.0
    with open('static/mau/' + config_filename + '.json') as f:
        config = json.load(f)
        scale_img = float(config['scale_img'])
        for rect in config['rectangles']:
            rectnames.append(rect['rectname'])
            rectangles.append(rect['rectangle'])

    image_process_filename = session['filename2']
    json_config_name = session['config_filename2']
    image_confile_filename = 'static/mau/' + json_config_name.replace('.json','') + '.png'

    _ = alignFile(image_confile_filename, image_process_filename)
    #print(a)
    return render_template('view2.html', filename = session['filename2'], list_config_files = session['list_config_files'], config_filename = session['config_filename2'], rectnames = rectnames, rectangles = rectangles, scale_img=scale_img)

@app.route('/post_align_<string:config_filename>',methods=["POST","GET"])
def postConfigAndAlign(config_filename):
    if 'filename2' not in session:
        return index2()
    session['config_filename2'] = config_filename
    config = []
    rectnames = []
    rectangles = []
    scale_img = 0.0
    with open('static/mau/' + config_filename + '.json') as f:
        config = json.load(f)
        scale_img = float(config['scale_img'])
        for rect in config['rectangles']:
            rectnames.append(rect['rectname'])
            rectangles.append(rect['rectangle'])

    image_process_filename = session['filename2']
    json_config_name = session['config_filename2']
    image_config_filename = 'static/mau/' + json_config_name.replace('.json','') + '.png'

    a = alignFile(image_config_filename, image_process_filename)
    session['filename2'] = a
    #print(a)
    resp = jsonify(success=True, rectnames=rectnames, rectangles=rectangles, scale_img=scale_img)
    if request.method == "POST":
        return resp
    if request.method == "GET":
        return render_template('view2.html', filename = session['filename2'], list_config_files = session['list_config_files'], config_filename = session['config_filename2'], rectnames = rectnames, rectangles = rectangles, scale_img=scale_img)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

@app.route('/recognize_all_rectangles2',methods=['POST'])
def recognizeAllrectangle2():
    rq = request.get_json()
    rectangles = rq['rectangles']
    scale_img = rq['scale_img']*0.6
    filename = session['filename2']
    img = cv2.imread(filename)
    texts = []
    for rectangle in rectangles:
        x = rectangle['x']/scale_img
        y = rectangle['y']/scale_img
        w = rectangle['width']/scale_img
        h = rectangle['height']/scale_img
        #print(rectangle['x'],rectangle['y'],rectangle['width'],rectangle['height'])
        crop_img = img[int(y):int(y+h), int(x):int(x+w)]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        #plt.imshow(crop_img)
        #plt.show()

        #_, crop_img = cv2.threshold(crop_img, kmeans(input_img=img, k=8, i_val=2)[0], 255, cv2.THRESH_BINARY)
        
        #plt.imshow(crop_img)
        #plt.show()
        #print(crop_img.shape)
        #crop = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        #crop = cv2.GaussianBlur(crop, (1, 1), 0)
        #_, crop_img = cv2.threshold(crop_img, 120 ,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # plt.imshow(crop)
        # plt.show()
        text = pytesseract.image_to_string(crop_img, config='-l vie --psm 13')
        texts.append(text)
    print(texts)
    resp = jsonify(success=True,data=texts)
    return resp

from preprocess_image import alignFile
@app.route('/preprocess_image', methods=["GET"])
def preprocessImage():
    print(session['config_filename2'])
    image_process_filename = session['filename2']
    json_config_name = session['config_filename2']
    image_confile_filename = 'static/mau/' + json_config_name.replace('.json','') + '.png'

    _ = alignFile(image_confile_filename, image_process_filename)

    rectnames = []
    rectangles = []
    scale_img = 0.0
    with open('static/mau/' + json_config_name + '.json') as f:
        config = json.load(f)
        scale_img = float(config['scale_img'])
        for rect in config['rectangles']:
            rectnames.append(rect['rectname'])
            rectangles.append(rect['rectangle'])

    return render_template('view2.html', filename = session['filename2'], list_config_files = session['list_config_files'], config_filename = session['config_filename2'], rectnames = rectnames, rectangles = rectangles, scale_img=scale_img)

if __name__ == '__main__':
    app.secret_key = 'dangvansam'
    #app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
    #app.run(debug=True, host="192.168.2.26", port=4040)