import cv2
import glob
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral","anger","contempt","disgust","fear","happy","sadness","surprise"]
def detect_faces(emotion):
    files = glob.glob("sorted_set\\%s\\*" %emotion)
    file_number=0
    for f in files:
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
                cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion,file_number),out)
            except:
                pass
        file_number+=1

for emotion in emotions:
    detect_faces(emotion)