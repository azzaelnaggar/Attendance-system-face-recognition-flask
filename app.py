from flask import Flask,render_template,Response
import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime

app=Flask(__name__)
cap=cv2.VideoCapture(0)
frame_resizing = 0.25
# from main.ipynb we have 2 lists the encoded images and classNames
with open('encoded_images.pkl', 'rb') as f:
   encodeListKnown = pickle.load(f)
# with open('classNames.pkl', 'rb') as f:
#    classNames = pickle.load(f)
classNames=['azza', 'Elon Musk']
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

    return("done")        

def generate_frames():
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), fx=frame_resizing, fy=frame_resizing)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
          
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
     
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
             
                faceLoc = np.array(faceLoc)
                faceLoc = faceLoc / 0.25
                faceLoc=faceLoc.astype(int)
                y1, x2, y2, x1 = faceLoc[0], faceLoc[1], faceLoc[2], faceLoc[3]
            
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)
        ret, jpeg = cv2.imencode('.jpg', img)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    # Set to global because we refer the video variable on global scope, 
		# Or in other words outside the function
    global video
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)