import cv2
capture = cv2.VideoCapture('D://Programming//FaceRecognition//test.mkv')
if (capture.isOpened() == False):
    print("Not able to open video")
    print(cv2.getBuildInformation())
while (capture.isOpened()):
    ret,frame = capture.read()
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()