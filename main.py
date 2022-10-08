import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8, maxHands=2)


while True:
    success,img = cap.read()
    hands, img = detector.findHands(img) # With Drawing of Hands
    #hands = detector.findHands(img,draw=False) # No Drawing of hand

    # Hand- dict (lmlist-bbox-center-type)
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] # List of 21 Landmarks points
        bbox1 = hand1["bbox"] # Bounding box info x,y,w,h
        centerPoint1 = hand1["center"] # center of the hand cx,cy
        handType1 = hand1["type"] # hand Type Left or Right

        # print(len(lmlist1),lmlist1)
        #print(bbox1)
        #print(centerPoint1)
        #print(handType1)

        fingers1 = detector.fingersUp(hand1)
        # length, info, img = detector.findDistance(lmlist1[8], lmlist1[12], img)

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmarks points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2["center"]  # center of the hand cx,cy
            handType2 = hand2["type"]  # hand Type Left or Right

            fingers2 = detector.fingersUp(hand2)

            length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)

            #print(fingers1,fingers2)



    cv2.imshow("Image",img)
    cv2.waitKey(1)