import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
#Declare values
emotions = ["neutral","anger","contempt","disgust","fear","happy","sadness","surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
#clf = KNeighborsClassifier()
def get_files(emotion):
    files = glob.glob("dataset\\%s\\*" %emotion)
    return files

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist) #Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #get distance between each point and the central point in both axes
        ycentral = [(y-ymean) for y in ylist]

        if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)

    if len(detections) < 1: 
        landmarks_vectorised = "error"
    return landmarks_vectorised

def make_sets():
    training_data = []
    training_labels = []
    for emotion in emotions:
        training = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                training_data.append(landmarks_vectorised) #append image array to training data list
                training_labels.append(emotions.index(emotion))
    return training_data, training_labels


#Main function
print("Making sets:")
training_data, training_labels = make_sets()
npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
npar_trainlabs = np.array(training_labels)
print("Training: %s images" %len(training_labels)) #train SVM
clf.fit(npar_train, training_labels)
cap = cv2.VideoCapture(0)
plt.rcdefaults()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Emotion Recognition",frame)
    clahe_image = clahe.apply(gray)
    landmarks_vectorised = get_landmarks(clahe_image)
    if landmarks_vectorised == "error":
        print "No face detected"
        pass
    else:
        scores = clf.predict_proba(landmarks_vectorised)
        """max = scores[0][0]
        vt = 0
        for i in range(0,8):
            if scores[0][i]>max:
                max=scores[0][i]
                vt = i
        print emotions[vt]"""
        for index in range(0,len(emotions)):
            scores[0][index]*=100
        y_pos = np.arange(len(emotions))
        plt.barh(y_pos,scores[0],align='center',alpha=0.5)
        plt.yticks(y_pos,emotions)
        plt.xlabel("%")
        plt.show()
        """for emotion in emotions:
            k = emotions.index(emotion)
            print emotion,": ",scores[0][k]*100"""
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()