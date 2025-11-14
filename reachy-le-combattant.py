

import cv2, math, time, threading, queue, json, os, sys
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# CONFIG

YOLO_MODEL = "yolov8n-pose.pt"
CONF_THR = 0.5

GAIN_Y = 30.0
MAX_Y = 25.0

ANT_MAX_DEG = 160.0

EMA = 0.28
DEADBAND_Y_MM = 0.3
DEADBAND_A_DEG = 0.5

MODEL_PATH = "vosk-model-small-en-us-0.15"   
SAMPLE_RATE = 16000
BLOCKSIZE = 8000

KEYWORDS = ["hit", "respect", "yes", "neck crack"]
COOLDOWN_S = 1.0

voice_q = queue.Queue()


# HELPERS


def clamp(v, lo, hi): return max(lo, min(hi, v))
def ema(prev, new, a): return new if prev is None else (1-a)*prev + a*new

# COCO indices
L_SH=5; R_SH=6
L_WR=9; R_WR=10
L_HIP=11; R_HIP=12

def valid(conf, idx):
    return idx < len(conf) and conf[idx] >= CONF_THR


# ANIMATIONS


def headbutt_fast(rm):
    print("⚔️  HEADBUTT!")
    rm.set_target(head=create_head_pose(pitch=-35, degrees=True))
    time.sleep(0.12)
    rm.set_target(head=create_head_pose(pitch=+35, degrees=True))
    time.sleep(0.12)
    rm.set_target(head=create_head_pose())

def salute(rm):
    print("🤵 RESPECT")
    rm.set_target(head=create_head_pose(pitch=-20, degrees=True))
    time.sleep(0.4)
    rm.set_target(head=create_head_pose())

def nod(rm):
    print("🙂 YES")
    for _ in range(2):
        rm.set_target(head=create_head_pose(pitch=-15, degrees=True))
        time.sleep(0.25)
        rm.set_target(head=create_head_pose(pitch=+10, degrees=True))
        time.sleep(0.25)
    rm.set_target(head=create_head_pose())

def neck_crack(rm):
    print("🦴 NECK CRACK (slow)")
    rm.set_target(head=create_head_pose(roll=40, degrees=True))
    time.sleep(0.5)
    rm.set_target(head=create_head_pose(roll=-40, degrees=True))
    time.sleep(0.5)
    rm.set_target(head=create_head_pose())
    time.sleep(0.2)


# VOICE THREAD


voice_command = None
last_trigger = 0.0

def audio_callback(indata, frames, time_info, status):
    voice_q.put(bytes(indata))

def listen_voice():
    global voice_command, last_trigger

    if not os.path.isdir(MODEL_PATH):
        print(f"[ERROR] Vosk model not found: {MODEL_PATH}")
        sys.exit(1)

    print(" Loading offline Vosk model…")
    model = Model(MODEL_PATH)

    rec = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(KEYWORDS))
    print("🎤 Listening for:", KEYWORDS)

    with sd.RawInputStream(samplerate=SAMPLE_RATE,
                           blocksize=BLOCKSIZE,
                           dtype="int16",
                           channels=1,
                           callback=audio_callback):
        while True:
            data = voice_q.get()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = res.get("text", "").strip().lower()
                if text:
                    print("🗣️ said:", text)
                    for k in KEYWORDS:
                        if k in text:
                            now = time.time()
                            if now - last_trigger >= COOLDOWN_S:
                                print("⚡ COMMAND:", k)
                                voice_command = k
                                last_trigger = now


# MARIONETTE MATH 


def hips_to_y(xy, conf, W):
    if not (valid(conf, L_SH) and valid(conf, R_SH)
            and valid(conf, L_HIP) and valid(conf, R_HIP)):
        return 0.0
    hip_mid_x = (xy[L_HIP][0] + xy[R_HIP][0])/2
    cx = W / 2
    shoulder_w = abs(xy[L_SH][0] - xy[R_SH][0])
    if shoulder_w < 1: shoulder_w = 1
    norm = (hip_mid_x - cx) / shoulder_w
    return clamp(norm*GAIN_Y, -MAX_Y, MAX_Y)

def wrists_to_antennas_no_calib(xy, conf):
    # use shoulder_y as reference
    if not (valid(conf, L_SH) and valid(conf, R_SH)):
        return None, None

    shoulder_y = (xy[L_SH][1] + xy[R_SH][1]) / 2.0
    if shoulder_y < 10:  # safety
        return None, None

    def one(wrist_idx, sign):
        if not valid(conf, wrist_idx): return None
        wrist_y = xy[wrist_idx][1]

        # higher wrist -> lower wrist_y -> larger numerator
        raw_t = (shoulder_y - wrist_y) / shoulder_y
        t = clamp(raw_t, 0.0, 1.0)  # always hits 1.0 when wrist high
        deg = (1.0 - t) * ANT_MAX_DEG
        return sign * deg

    aR = one(R_WR, -1)   # right antenna: -160..0
    aL = one(L_WR, +1)   # left antenna:  0..+160
    return aR, aL


# MAIN


def main():
    global voice_command

    # Voice thread
    vt = threading.Thread(target=listen_voice, daemon=True)
    vt.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    y_s = None
    aR_s = None
    aL_s = None

    model = YOLO(YOLO_MODEL)

    print("READY! Move to control robot, or say:")
    print("   'hit' → headbutt")
    print("   'respect' → salute bow")
    print("   'yes' → nod twice")
    print("   'neck crack' → slow +40°, -40° roll")

    with ReachyMini() as rm:
        while True:
            ok, frame = cap.read()
            if not ok: break
            H, W = frame.shape[:2]

            # Voice actions
            if voice_command == "hit":
                headbutt_fast(rm)
                voice_command = None

            elif voice_command == "respect":
                salute(rm)
                voice_command = None

            elif voice_command == "yes":
                nod(rm)
                voice_command = None

            elif voice_command == "neck crack":
                neck_crack(rm)
                voice_command = None

            # YOLO control
            res = model.predict(frame, imgsz=640, conf=CONF_THR, verbose=False)[0]

            if res.keypoints:
                kp = max(res.keypoints,
                         key=lambda k: float(k.conf.mean() if k.conf is not None else 0))
                xy = kp.xy[0].cpu().numpy()
                conf = kp.conf[0].cpu().numpy()

                # hips -> head-Y
                y_raw = hips_to_y(xy, conf, W)

                # wrists -> antennas (new method, no denom)
                aR_raw, aL_raw = wrists_to_antennas_no_calib(xy, conf)

                if aR_raw is None: aR_raw = aR_s if aR_s is not None else -ANT_MAX_DEG
                if aL_raw is None: aL_raw = aL_s if aL_s is not None else +ANT_MAX_DEG

                # deadband
                if y_s is not None and abs(y_raw - y_s) < DEADBAND_Y_MM: y_raw = y_s
                if aR_s is not None and abs(aR_raw - aR_s) < DEADBAND_A_DEG: aR_raw = aR_s
                if aL_s is not None and abs(aL_raw - aL_s) < DEADBAND_A_DEG: aL_raw = aL_s

                # smooth
                y_s = ema(y_s, y_raw, EMA)
                aR_s = ema(aR_s, aR_raw, EMA)
                aL_s = ema(aL_s, aL_raw, EMA)

                rm.set_target(
                    head=create_head_pose(y=float(y_s), mm=True),
                    antennas=[math.radians(float(aR_s)), math.radians(float(aL_s))]
                )

            small = cv2.resize(res.plot() if res else frame, (640, 640))
            cv2.imshow("Reachy Le combattant", small)
            if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
