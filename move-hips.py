import math, cv2, time
from ultralytics import YOLO
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

model = YOLO("yolov8n-pose.pt")

GAIN = 30     
CONF = 0.5

# keypoints indices
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12

def valid(conf, idx): return idx < len(conf) and conf[idx] >= CONF

cap = cv2.VideoCapture(0)

with ReachyMini() as rm:
    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]

        res = model.predict(frame, imgsz=640, conf=CONF, verbose=False)[0]

        if res.keypoints:
            kp = max(res.keypoints, key=lambda k: float(k.conf.mean()))
            xy = kp.xy[0].cpu().numpy()
            conf = kp.conf[0].cpu().numpy()

            if valid(conf, L_SH) and valid(conf, R_SH) and valid(conf, L_HIP) and valid(conf, R_HIP):

                # milieu des hanches
                hip_mid_x = (xy[L_HIP][0] + xy[R_HIP][0]) / 2.0

                # centre de l’image
                frame_center_x = W / 2.0

                # normalisation avec largeur épaules
                shoulder_width = abs(xy[L_SH][0] - xy[R_SH][0])
                if shoulder_width < 1: shoulder_width = 1

                # déplacement horizontal normalisé
                norm = (hip_mid_x - frame_center_x) / shoulder_width

                # mouvement en mm sur l’axe Y
                y_move = norm * GAIN     # mm
                # tu peux clipper un peu
                y_move = max(-20, min(20, y_move))

                # envoyer au robot
                rm.set_target(head=create_head_pose(y=y_move, mm=True))

        cv2.imshow("pose", res.plot())
        if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
