import cv2
import mediapipe as mp
import time
import numpy as np
import yolov5
model = yolov5.load('keremberke/yolov5m-garbage')
print("Model Uploaded")
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

import streamlit as st
import time as time
from PIL import Image
import sys
import base64

# file_ = open("C:/Users/manvi/Desktop/Hacklytics/TrashNDash_Logo_transbg.gif", "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")
# file_.close()

logo = Image.open('C:/Users/manvi/Desktop/Hacklytics/TrashNDash_Logo_transbg.gif')

_left, mid, _right = st.columns(3)
with mid:
   st.image(logo)

# st.markdown(
#     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
#     unsafe_allow_html=True,
# )

def directtobin(x):
    print('stl',x)
    yellow = Image.open('C:/Users/manvi/Desktop/Hacklytics/yellow.jpg')
    red = Image.open('C:/Users/manvi/Desktop/Hacklytics/red.jpg')
    blue = Image.open('C:/Users/manvi/Desktop/Hacklytics/blue.jpg')
    green = Image.open('C:/Users/manvi/Desktop/Hacklytics/green.jpg')
    purple = Image.open('C:/Users/manvi/Desktop/Hacklytics/purple.jpg')

    col1, col2, col3, col4, col5 = st.columns(5)
    placeholder = st.empty()
    if(x==1):
        with col1:
            placeholder = st.image(yellow, use_column_width=True)
    elif(x==2):
        with col2:
            placeholder = st.image(red, use_column_width=True)
    elif(x==3):
        with col3:
            placeholder = st.image(blue, use_column_width=True)
    elif(x==4):
        with col4:
            placeholder = st.image(green, use_column_width=True)
    elif(x==5):
        with col5:
            placeholder = st.image(purple, use_column_width=True)
    time.sleep(1.5)
    placeholder.empty()
# directtobin(4)
# x = sys.argv[1]
# directtobin(int(x))
# while(1):
#     x = int(input())
#     directtobin(x)

#cap = cv2.VideoCapture('videos/a.mp4')
pTime = 0

model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 4  # maximum number of detections per image
def get_class(img):


    # set image
    #img = 'CAN.png'
    #img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # perform inference
    img=cv2.resize(img,(640,640))
    results = model(img, size=640)

    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    # boxes = predictions[:, :4] # x1, y1, x2, y2
    # scores = predictions[:, 4]
    categories = predictions[:, 5]
    if(len(categories)>0):
        q = categories[0]+1 
        if q==1: directtobin(3) #biodegradable
        elif q==2: directtobin(2) #glass and metal
        elif q==3: directtobin(2) #glass and metal
        elif q==4: directtobin(1) #paper
        elif q==5: directtobin(5) #others
        elif q==6: directtobin(4) #plastic
        directtobin(min(int(categories[0])+1,4))
    print(predictions)
    # show detection bounding boxes on image
    #results.show()

    # save results into "results/" folder
    #results.save(save_dir='results/')

count=0
while True:
    count+=1
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    x1,y1,x2,y2=0,0,0,0
    x1_a=[]
    y1_a=[]
    x2_a=[]
    y2_a=[]
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w,c = img.shape
            #print(id, lm)
            if(id==20 or id==22 or id==18 or id ==16):
                x1_a.append(lm.x*w)
                y1_a.append(lm.x*h)
                 
            if(id==21 or id==15 or id==19 or id ==17):
                x2_a.append(lm.x*w)
                y2_a.append(lm.x*h)
                

            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    if((len(x1_a)*len(x2_a)*len(y1_a)*len(y2_a))!=0):
        x1=int(np.mean(x1_a))
        x2=int(np.mean(x2_a))
        y1=int(np.mean(y1_a))
        y2=int(np.mean(y2_a))
        y=int((y1+y2)/2)
        diff=int(abs(x1-x2)/2)
        cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        #print(count)
        if(diff>40 and diff<150 and count>=50):
            #print(count)
            get_class(imgRGB[max(0,y-diff):min(y+diff,h), x1:x2])
            count=0
        #print(x1,x2,y1,y2,y)
        # Naming a window
        #cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        #img=cv2.resize(img,(250,250))
        # Using resizeWindow()
        #cv2.resizeWindow("Resized_Window", 100, 100)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    else:
        cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        #print("NO CROP")

        cv2.imshow("Image", img)
        cv2.waitKey(1)




