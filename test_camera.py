import cv2

cap = cv2.VideoCapture(2)
print(cap)

while True:
    r, frame = cap.read()
    if r:
        cv2.imshow("prev", frame)
        
        cv2.waitKey(1)
        
        # if k == 27:
        #     break 