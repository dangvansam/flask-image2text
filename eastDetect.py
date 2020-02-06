from PIL import Image
import pytesseract
import argparse
import cv2
import cv2 as cv
import os
import numpy as np

############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []
    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]
            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue
            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def kmeans(input_img, k, i_val):
    hist = cv2.calcHist([input_img],[0],None,[256],[0,256])
    img = input_img.ravel()
    img = np.reshape(img, (-1, 1))
    img = img.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(img,k,None,criteria,10,flags)
    centers = np.sort(centers, axis=0)

    return centers[i_val].astype(int), centers, hist

#nối các block gần nhau chung 1 line
import matplotlib.pyplot as plt
import math
def checkSameLine(block1,block2,thr):
    b0_x = block1[0][0]
    b0_y = block1[0][1]
    b1_x = block1[1][0]
    b1_y = block1[1][1]
    b2_x = block1[2][0]
    b2_y = block1[2][1]
    b3_x = block1[3][0]
    b3_y = block1[3][1]
    #print(b0_x,b0_y)
    a0_x = block2[0][0]
    a0_y = block2[0][1]
    a1_x = block2[1][0]
    a1_y = block2[1][1]
    a2_x = block2[2][0]
    a2_y = block2[2][1]
    a3_x = block2[3][0]
    a3_y = block2[3][1]

    plt.plot(b0_x,b0_y,'bo',label='0')
    plt.plot(b1_x,b1_y,'bo',label='1')
    plt.plot(b2_x,b2_y,'bo',label='2')
    plt.plot(b3_x,b3_y,'bo',label='3')
    plt.plot(a0_x,b0_y,'ro',label='0')
    plt.plot(a1_x,b1_y,'ro',label='1')
    plt.plot(a2_x,a2_y,'ro',label='2')
    plt.plot(a3_x,a3_y,'ro',label='3')
    #plt.show()
    new_block = np.zeros((4,2))
    #print(new_block)
    if math.sqrt(math.pow((b1_y - a1_y),2)) <= thr:
        if b0_x < a0_x:
            new_block[0][0] = b0_x
            new_block[0][1] = b0_y
            new_block[1][0] = b1_x
            new_block[1][1] = b1_y
            new_block[2][0] = a2_x
            new_block[2][1] = a2_y
            new_block[3][0] = a3_x
            new_block[3][1] = a3_y
        else:
            new_block[0][0] = a0_x
            new_block[0][1] = a0_y
            new_block[1][0] = a1_x
            new_block[1][1] = a1_y
            new_block[2][0] = b2_x
            new_block[2][1] = b2_y
            new_block[3][0] = b3_x
            new_block[3][1] = b3_y
        #print(new_block)
    return new_block

def connectBlock(list_block = [], use = True, thr = 2):
    if use == True:
        l = len(list_block)
        #print(l)
        for i in range(0,l):
            #print(i)
            b1 = list_block[i]
            for j in range(i+1,l):
                #print(j)
                b2 = list_block[j]
                #print(checkSameLine(b1,b2)[0][0])
                if checkSameLine(block1 = b1,block2= b2,thr = thr)[0][0] > 0:
                    list_block[i] = checkSameLine(block1 = b1,block2= b2,thr = thr)
                    l-=1
                    for k in range(j,l):
                        list_block[k] = list_block[k+1]
                        #print(l)
        #print(l)
        return list_block[:l]
    else:
        return list_block

def detect(filename, scale_img):
    confThreshold = 0.5
    nmsThreshold = 0.5
    model = 'frozen_east_text_detection.pb'
    # Load network
    net = cv.dnn.readNet(model)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")
    #print(args.input)
    # Open a video file or an image file or a camera stream
    #cap = cv.VideoCapture(args.input if args.input else 0)
    #cap = cv.VideoCapture('CO_D.png')
    #while cv.waitKey(1) < 0:
        # Read frame
        #hasFrame, frame = cap.read()
        # if not hasFrame:
        #     cv.waitKey()
            #break
    #input_ = 'cmnd/CMND_5.png' #ảnh đầu vào để detect text.
    input_ = filename
    frame = cv.imread(input_)
    #print(frame.shape)
    # Get frame height and width
    frame = frame[:,:,:3]
    ##########################################
    #nhận dạng trực tiếp file ảnh bằng tesseract không qua block
    frame_cp = frame.copy()
    img_gray0 = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2GRAY)
    img_gray0 = cv2.medianBlur(img_gray0,3)
    #img_gray0 = cv2.adaptiveThreshold(img_gray0,255,1,1,11,2)
    img_gray0 = unsharp_mask(img_gray0)
    #text_block0 = pytesseract.image_to_string(img_gray0,lang='vie')
    #cv.imshow('no detect',img_gray0)
    #cv.waitKey()
    #print(text_block0)
    ##########################################
    #nhận dạng từng block text rồi dùng tesseract để nhận dạng text trên block đó
    frame_org = frame.copy()
    height_ = frame.shape[0]
    width_ = frame.shape[1]
    print(frame.shape)
    print(width_, height_)
    inpHeight, inpWidth = int(height_ / 32)*32, int(width_ / 32)*32
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)

    print(inpWidth, inpHeight)
    # Create a 4D blob from frame.
    blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

    # Run the model
    net.setInput(blob)
    output = net.forward(outputLayers)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

    scores = output[0]
    geometry = output[1]
    [boxes, confidences] = decode(scores, geometry, confThreshold)
    indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
    vertices_all = []
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv.boxPoints(boxes[i[0]])
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH
        vertices_all.append(vertices)
    new_vertices = connectBlock(vertices_all, use = True, thr = 3)
    rectangles = []
    rectnames = []
    for count,i in enumerate(new_vertices):
        rectangle = {}
        vertices = np.int0(i)
        #print(vertices)
        cropped = frame_org[int(vertices[1][1])-5:int(vertices[0][1])+5,int(vertices[1][0])-5:int(vertices[2][0])+5]
        #if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            #continue
        x = int(vertices[1][0]*scale_img)
        y = int(vertices[1][1]*scale_img)
        w = int((vertices[2][0] - vertices[1][0])*scale_img)
        h = int((vertices[0][1] - vertices[1][1])*scale_img)
        rectangle['x'] = x
        rectangle['y'] = y
        rectangle['width'] = w
        rectangle['height'] = h
        #print(rectangle)
        rectangles.append(rectangle)
        rectnames.append('rectname'+str(count))
    #print(rectangles, len(rectangles))
    return rectangles, rectnames

#detect('static/uploaded/CMND_1.png')