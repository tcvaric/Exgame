import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#Inmort imges
imgBackground = cv2.imread("Image/bord.png")
imgGameover = cv2.imread("Image/basket.png")
imgBall = cv2.imread("Image/basketball.png", cv2.IMREAD_UNCHANGED)
imgleft = cv2.imread("Image/left.png", cv2.IMREAD_UNCHANGED)
imgright = cv2.imread("Image/right.png", cv2.IMREAD_UNCHANGED)

detector = HandDetector(detectionCon=0.8, maxHands=2)
ballPos = [150, 100]
speedX = 20
speedY = 15
gameover = False
score = [0, 0]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    #imgRaw = img.copy()

    #hands, img = detector.findHands(img, flipType=False)
    hands = detector.findHands(img, flipType=False, draw=False)

    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    #Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgright.shape
            y1 = y - h1//2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgright, (40, y1))
                if 40 < ballPos[0] < 40 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgleft, (1160, y1))
                if 1100 < ballPos[0] < 1100 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1

    if ballPos[0] < 30 or ballPos[0] > 1180:
        gameover = True

    if gameover:
        img = imgGameover
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (891, 785), cv2.FONT_HERSHEY_COMPLEX,
                    5, (173, 255, 47), 6)

    #If game not over move the ball
    else:
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY
        ballPos[0] += speedX
        ballPos[1] += speedY

        img = cvzone.overlayPNG(img, imgBall, ballPos)

        cv2.putText(img, str(score[0]), (300, 600), cv2.FONT_HERSHEY_COMPLEX, 3, (173, 255, 47), 5)
        cv2.putText(img, str(score[1]), (900, 600), cv2.FONT_HERSHEY_COMPLEX, 3, (173, 255, 47), 5)

    #img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    Key = cv2.waitKey(1)
    if Key == ord('r'):
        ballPos = [150, 100]
        speedX = 20
        speedY = 15
        gameover = False
        score = [0, 0]
        imgGameover = cv2.imread("Image/basket.png")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()