import numpy as np
import cv2
import glob
import threading


emotions = ["neutral","anger","contempt","disgust","fear","happy","sadness","surprise"]
fisherface = cv2.createFisherFaceRecognizer()

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def get_files(emotion):
    files = glob.glob("dataset\\%s\\*" %emotion)
    return files

def detect_face(gray):
    face= faceDet.detectMultiScale(gray,1.3,5)
    
    out = ""
    if len(face) == 1:
        faceFeatures =face
    
    else:
        faceFeatures=""
    for (x,y,w,h) in faceFeatures:
        gray = gray[y:y+h, x:x+w]
        try:
            out = cv2.resize(gray, (350,350))
        except:
            pass
    return out

def make_sets():
    training_data = []
    training_labels = []
    for emotion in emotions:
        training = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

    return training_data,training_labels

def train_data():
    training_data,training_labels=make_sets()

    print("Training fisher face classifier")
    print("size of training set is: ", len(training_labels)," images")

    fisherface.train(training_data,np.asarray(training_labels))
def run_recognizer(gray):

    frame = detect_face(gray)
    if frame=="":
        print("No face detected")
        return
    pred,conf=fisherface.predict(frame)
    print(emotions[pred])

    

train_data()
cap = cv2.VideoCapture('test.mkv')

while(True):
    ret,frame = cap.read()   

    cv2.imshow('Emotion Recognition',frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    run_recognizer(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

