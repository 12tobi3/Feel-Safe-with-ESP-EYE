import face_recognition
import cv2
import numpy as np
import random
import urllib.request

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
##urllib.request.urlopen("http://172.20.10.6/control?var=led_intensity&val=0").read()

#urllib.request.urlopen("http://172.20.10.6/control?var=vflip&val=1").read()

URL = "http://172.20.10.6"
i=0
# Face recognition and opencv setup
video_capture = cv2.VideoCapture(URL + ":81/stream")
# For Webcam
# video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
chris_image = face_recognition.load_image_file("/Users/christian.willmann/Desktop/Master/3. Semester/Wissensmana./FacePics/Chris.jpg")
chris_face_encoding = face_recognition.face_encodings(chris_image)[0]




eye_cascade = cv2.CascadeClassifier('/Users/christian.willmann/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml')



# Create arrays of known face encodings and their names
known_face_encodings = [
    chris_face_encoding
]
known_face_names = [
    "Chris"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            face_names.append(name)

    process_this_frame = not process_this_frame

    eyes = eye_cascade.detectMultiScale(rgb_small_frame)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,(ex*4,ey*4),(ex*4+ew*4,ey*4+eh*4),(0,255,0),2)
        cv2.imshow("Eyes",frame)


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if (name == "Unknown") & (i>20):
            #urllib.request.urlopen("http://172.20.10.6/control?var=led_intensity&val=255").read()
            cv2.imwrite("/Users/christian.willmann/Desktop/Master/3. Semester/Wissensmana./testBilder/FaceDetectionRndmNr_"+str(random.randint(0,999))+str(random.randint(0,999))+".jpeg", frame)
            i=0
        print(i)
        i=i+1
        #urllib.request.urlopen("http://172.20.10.6/control?var=led_intensity&val=0").read()


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
