import os
import cv2
from imutils.video import VideoStream
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import datetime
from speech_to_text import STT
stt = STT()
from speech import  speak


# Returns the current local date
today = datetime.datetime.today().strftime('%m/%d/%Y')
exp="01/01/2022"

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")



def Create_id():
    NAME=stt.speech_to_text()
    print("*******************")
    result = DeepFace.analyze("opencv_frame.jpg")
    print("*******************")
    print(result)
    AGE = result['age']
    Nation = result["dominant_race"]
    Gender = result["gender"]




    im1 = Image.open('./static/images/1.jpg')
    im2 = Image.open('faces.jpg')
    im3=Image.open('./static/images/2.jpg')
    back_im = im1.copy()


    font = ImageFont.truetype('arial.ttf', 30)
    font1 = ImageFont.truetype('arial.ttf', 20)
    draw = ImageDraw.Draw(back_im)
    draw.text(xy=(95, 445), text='{}'.format(NAME), fill=(0, 0, 0), font=font1)
    draw.text(xy=(485, 482), text='{}'.format(AGE), fill=(255, 0, 0), font=font)
    draw.text(xy=(530, 392), text='{}'.format(Gender), fill=(255, 0, 0), font=font)
    draw.text(xy=(580, 315), text='{}'.format(Nation), fill=(255, 0, 0), font=font)

    im2 = im2.resize((180, 180))
    back_im.paste(im2, (60, 240))
    print(im2.size)
    back_im.save('./static/images/img/output1.jpg', quality=95)



    draw2 = ImageDraw.Draw(im3)
    draw2.text(xy=(410, 155), text='{}'.format(today), fill=(255, 0, 0), font=font)
    draw2.text(xy=(420, 235), text='{}'.format(exp), fill=(255, 0, 0), font=font)
    im3.save('./static/images/img/output2.jpg', quality=95)

    os.remove("opencv_frame.jpg")
    os.remove("faces.jpg")



def Predict_image():
    speak("I'm taking your image smile please")
    vs = VideoStream(0).start()
    image = vs.read()
    img_name = "opencv_frame.jpg"
    cv2.imwrite(img_name, image)
    frame = cv2.imread("opencv_frame.jpg")
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #to save face image
        roi_color = frame[y:y + h, x:x + w]
        cv2.imwrite('faces.jpg', roi_color)
    Create_id()


cv2.destroyAllWindows()