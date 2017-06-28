import cv2, math, numpy as np, dlib
from sklearn.externals import joblib
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

warnings.filterwarnings("ignore")
#Declare values
emotions = ["neutral","anger","fear","disgust","happy","sadness","surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = joblib.load('training_model.pkl')

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections):
        shape = predictor(image, d)
        for i in range(1,68): #There are 68 landmark points on each face
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
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



#Main function

cap = cv2.VideoCapture('test.mkv')
plt.rcdefaults()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
total_frame = 0
detection_frame=0
total_scores = [0]*len(emotions)
def animate(i):
    global total_frame, total_scores,detection_frame
    ret, frame = cap.read()
    
    cv2.imshow("Video Recognition",frame)
    total_frame+=1
    if total_frame%20==0:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        landmarks_vectorised = get_landmarks(clahe_image)
        if landmarks_vectorised == "error":
            return
        else:
            detection_frame+=1
            scores = clf.predict_proba(landmarks_vectorised)
            temp = scores[0][len(emotions)-1]
            scores[0][len(emotions)-1]=scores[0][0]
            scores[0][0]=temp
            max = scores[0][0]
            vt = 0                
            for index in range(0,len(emotions)):
                if scores[0][index]>max:
                    max=scores[0][index]
                    vt = index
                scores[0][index]*=100
                total_scores[index]+=scores[0][index]
            y_pos = np.arange(len(emotions))
            ax1.clear()
            barlist = plt.barh(y_pos,scores[0],align='center',alpha=0.5)
            plt.yticks(y_pos,emotions)
            plt.xlabel("%")
            barlist[vt].set_color('r')
while (True):
    ani = animation.FuncAnimation(fig,animate,interval=50)
    plt.show()
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
cls()
for emotion in emotions:
    print("%s : %.2f"%(emotion,total_scores[emotions.index(emotion)]/detection_frame))
cap.release()
cv2.destroyAllWindows()
