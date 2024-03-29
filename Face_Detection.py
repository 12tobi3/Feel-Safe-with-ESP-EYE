
import cv2
import numpy as np
import time
import requests
import random


'''
INFO SECTION
- if you want to monitor raw parameters of ESP32CAM, open the browser and go to http://192.168.x.x/status
- command can be sent through an HTTP get composed in the following way http://192.168.x.x/control?var=VARIABLE_NAME&val=VALUE (check varname and value in status)
'''

# ESP32 URL
URL = "http://172.20.10.6"
AWB = True

# Face recognition and opencv setup
cap = cv2.VideoCapture(URL + ":81/stream")
face_classifier = cv2.CascadeClassifier('/Users/christian.willmann/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
face_classifier2 = cv2.CascadeClassifier('/Users/christian.willmann/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml') # insert the full path to haarcascade file if you encounter any problem # insert the full path to haarcascade file if you encounter any problem
eye_cascade = cv2.CascadeClassifier('/Users/christian.willmann/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml')
eye_left_cascade = cv2.CascadeClassifier('/Users/christian.willmann/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_lefteye_2splits.xml')
eye_right_cascade = cv2.CascadeClassifier('/Users/christian.willmann/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_righteye_2splits.xml')



def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:
        print("SET_QUALITY: something went wrong")
    return awb

if __name__ == '__main__':
    set_resolution(URL, index=8)
    i=0
    GOT=0
    while True:
        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                faces = face_classifier.detectMultiScale(gray)
                for (x, y, w, h) in faces:
                    img= cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 4)
                    crop_img = img[y:y+h, x:x+w]
                    



                    face2=face_classifier2.detectMultiScale(crop_img)
                    for (x, y, w, h) in face2:
                            faceCroppedPic=cv2.rectangle(crop_img, (x, y), (x + w, y + h), (255, 255, 0), 4)


                            eyes = eye_cascade.detectMultiScale(faceCroppedPic)
                            found_both=False
                            for (ex,ey,ew,eh) in eyes:
                                imShowPic=cv2.rectangle(faceCroppedPic,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                                found_both=True
                                cv2.imshow("test",imShowPic)





                            eye_left = eye_left_cascade.detectMultiScale(faceCroppedPic)
                            found_left=False
                            for (ex,ey,ew,eh) in eye_left:
                                cv2.rectangle(faceCroppedPic,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                                found_left=True




                            eye_right = eye_right_cascade.detectMultiScale(faceCroppedPic)
                            found_right=False
                            for (ex,ey,ew,eh) in eye_right:
                                cv2.rectangle(faceCroppedPic,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                                found_right=True
                            



                            if(found_left&found_right&found_both&(i>2)):
                                
                                cv2.imwrite("/Users/christian.willmann/Desktop/testBilder/FaceDetectionRndmNr_"+str(random.randint(0,999))+str(random.randint(0,999))+".jpeg", faceCroppedPic)
                                i=0
                            print(i)
                            i=i+1
            cv2.imshow("frame", frame)

            key = cv2.waitKey(1)

            if key == ord('r'):
                idx = int(input("Select resolution index: "))
                set_resolution(URL, index=idx, verbose=True)

            elif key == ord('q'):
                val = int(input("Set quality (10 - 63): "))
                set_quality(URL, value=val)

            elif key == ord('a'):
                AWB = set_awb(URL, AWB)

            elif key == ord('b'):
                break

    cv2.destroyAllWindows()
    cap.release()
