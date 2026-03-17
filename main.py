import cv2
import numpy as np
import time

# Load model
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus",
           "car","cat","chair","cow","diningtable","dog","horse","motorbike",
           "person","pottedplant","sheep","sofa","train","tvmonitor"]

cap = cv2.VideoCapture("videos/test.mp4")

prev_gray = None
prev_time = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # ---------- FIX ZOOM ISSUE ----------
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

    h, w = frame.shape[:2]

    # ---------- RESTRICTED AREA (BOTTOM RIGHT) ----------
    restricted_x1 = int(w * 0.60)
    restricted_y1 = int(h * 0.60)
    restricted_x2 = int(w * 0.85)
    restricted_y2 = int(h * 0.90)

    cv2.rectangle(frame,(restricted_x1,restricted_y1),
                  (restricted_x2,restricted_y2),(0,0,255),2)

    cv2.putText(frame,"Restricted Area",
                (restricted_x1,restricted_y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    # ---------- MOTION DETECTION ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    motion_text = "No movement"

    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        motion = np.sum(diff)

        if motion > 500000:
            motion_text = "Movement detected"

    prev_gray = gray

    # ---------- OBJECT DETECTION ----------
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.5:

            idx = int(detections[0,0,i,1])
            label = CLASSES[idx]

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")

            cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)

            activity = label

            if label == "person" and motion_text == "Movement detected":
                activity = "Person walking"

            if label == "car" and motion_text == "Movement detected":
                activity = "Car passing"

            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)

            # ---------- ALERT ----------
            if label == "person":
                if (restricted_x1 < centerX < restricted_x2) and (restricted_y1 < centerY < restricted_y2):

                    activity = "⚠ Suspicious Person"

                    cv2.putText(frame,"⚠ ALERT!",
                                (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,(0,0,255),4)

            cv2.putText(frame,activity,
                        (startX,startY-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    # ---------- COOL UPGRADE : FPS DISPLAY ----------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(frame,f"FPS: {int(fps)}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,0),2)

    cv2.imshow("SceneSense AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()  