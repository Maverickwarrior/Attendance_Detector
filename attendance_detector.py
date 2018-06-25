#importing the libraries
import numpy as np
import cv2

#importing the cascade xml file for face detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#loading the input image 
frame= cv2.imread("people.jpg")

#converting the colored image into gray-scale
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#detecting faces
faces= face_cascade.detectMultiScale(gray)

#drawing rectangles on detected faces
if len(faces)==0:
    print("No faces found")
else:
    print("Number of faces detected : " + str(faces.shape[0]) )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
    cv2.rectangle(frame,(0,frame.shape[0] - 190),(1425,frame.shape[0]),(255,255,255),-1)
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame,"Total strength:30 " + "Present: " + str(faces.shape[0]),(0,frame.shape[0]-70),font,3,(0,0,0),2,cv2.LINE_AA )
    
    cv2.imshow("Attendance",frame)
    cv2.imwrite("output4.jpg",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    