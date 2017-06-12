import glob
import cv2
import numpy as np
emotions = ["neutral","anger","contempt","disgust","fear","happy","sadness","surprise"]
fisherface = cv2.createFisherFaceRecognizer()

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

def get_files(emotion):
    files = glob.glob("dataset\\%s\\*" %emotion)
    return files

def detect_face():
    predict_files = glob.glob("predict\\*")
    for f in predict_files:
        frame = cv2.imread(f)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face= faceDet.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)
        face2= faceDet2.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)
        face3= faceDet3.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)
        face4= faceDet4.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)

        if len(face) == 1:
            faceFeatures =face
        elif len(face2)==1:
            faceFeatures = face2
        elif len(face3)==1:
            faceFeatures = face3
        elif len(face4)==1:
            faceFeatures = face4
        else:
            faceFeatures=""
        for (x,y,w,h) in faceFeatures:
            print "face found in file: %s" %f
            gray = gray[y:y+h, x:x+w]
            try:
                out = cv2.resize(gray, (350,350))
                cv2.imwrite("predict_data\k.png",out)
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

def run_recognizer():
    training_data,training_labels=make_sets()

    print "Training fisher face classifier"
    print "size of training set is: ", len(training_labels)," images"

    fisherface.train(training_data,np.asarray(training_labels))
    print "Predicting classification set"
    #prediction_data = glob.glob("predict_data//*")
    #for item in prediction_data:
    frame = detect_face()
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    pred,conf=fisherface.predict(frame)
    print emotions[pred]

detect_face()
run_recognizer()
    