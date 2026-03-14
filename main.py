"""
main.py
- YOLOv8 detection (ultralytics)
- Center trapezoid ROI trigger + approach detection
- Gesture-to-speech tied to a PERSON id (MediaPipe Hands + association)
- Facial expression detection (Haarcascade smile) with speech on change
- Emergency sound detection: Siren/Alarm (DANGER) vs Loud Impact (BE CAREFUL)
- Direction awareness (left / mid / right occupancy) + spoken "clear side" guidance
- TTS alerts (pyttsx3)
"""
# Note: this is a simple demo for proof-of-concept. In a real assistive device, you would want to optimize the model, use a more robust tracking method, and carefully design the alerting logic to avoid overwhelming the user.
import cv2, numpy as np, time, threading, collections, math, queue, sys
from ultralytics import YOLO
import mediapipe as mp
import pyttsx3
import sounddevice as sd

# ----------------- CONFIG -----------------
MODEL_NAME = "yolov8n.pt"
CONF_THRESH = 0.35

ROI_SCALE_W = 0.6
ROI_SCALE_H = 0.45

ALERT_CLASSES = ['person','bicycle','car','motorbike','bus','truck','dog','cat','cell phone']
APPROACH_AREA_GROWTH = 1.2
AREA_MEMORY = 6
ALERT_COOLDOWN = 1.8
SPEECH_ENABLED = True

# Sound detection
AUDIO_SAMPLERATE = 16000
AUDIO_BLOCK_SECONDS = 1.0
SOUND_RMS_THRESHOLD = 0.08
SIREN_FREQ_LOW = 400.0
SIREN_FREQ_HIGH = 3000.0
SPECTRAL_ENERGY_RATIO_THRESH = 0.2

# Gestures
GESTURE_SPEECH_COOLDOWN = 2.0

# Face/emotion
SMILE_DETECT_MIN_NEIGHBORS = 15

# Direction advisor
DIRECTION_SPEECH_COOLDOWN = 3.0
MIN_OBJECT_AREA_TO_BLOCK = 3000   # ignore tiny detections when deciding if a lane is blocked

# ------------------------------------------

# COCO names (80)
COCO_NAMES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
    "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

print("Loading YOLO model...")
model = YOLO(MODEL_NAME)

engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

def speak(text):
    if not SPEECH_ENABLED: return
    def _s():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=_s, daemon=True).start()

last_alerts = {}
lock_alerts = threading.Lock()
last_gesture_time = 0.0
gesture_last_spoken = None
last_direction_speech = 0.0
last_direction_phrase = None

# ----------------- SIMPLE TRACKER -----------------
class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}  # id -> {center, area_hist, label, last_seen, xyxy}

    def update(self, detections, now_ts):
        results = []
        used_ids = set()
        feats = []
        for d in detections:
            x1,y1,x2,y2 = d['xyxy']
            cx = (x1 + x2)/2; cy = (y1 + y2)/2
            area = max(1.0, (x2-x1) * (y2-y1))
            feats.append({'d':d, 'cx':cx, 'cy':cy, 'area':area})

        for f in feats:
            best_id, best_dist = None, 1e9
            for oid, obj in self.objects.items():
                if oid in used_ids: continue
                ox, oy = obj['center']
                dist = math.hypot(ox - f['cx'], oy - f['cy'])
                if dist < best_dist:
                    best_dist, best_id = dist, oid
            if best_id is None or best_dist > 80:
                oid = self.next_id; self.next_id += 1
                dq = collections.deque(maxlen=AREA_MEMORY); dq.append(f['area'])
                self.objects[oid] = {'center':(f['cx'],f['cy']), 'area_hist':dq,
                                     'label':f['d']['label'], 'last_seen': now_ts, 'xyxy': f['d']['xyxy']}
                used_ids.add(oid)
                results.append({'id':oid, 'xyxy':f['d']['xyxy'], 'label':f['d']['label'],
                                'area_hist': self.objects[oid]['area_hist']})
            else:
                oid = best_id
                self.objects[oid]['center'] = (f['cx'], f['cy'])
                self.objects[oid]['area_hist'].append(f['area'])
                self.objects[oid]['label'] = f['d']['label']
                self.objects[oid]['xyxy']  = f['d']['xyxy']
                self.objects[oid]['last_seen'] = now_ts
                used_ids.add(oid)
                results.append({'id':oid, 'xyxy':f['d']['xyxy'], 'label':f['d']['label'],
                                'area_hist': self.objects[oid]['area_hist']})
        for oid in [oid for oid,obj in self.objects.items() if now_ts - obj['last_seen'] > 1.5]:
            del self.objects[oid]
        return results

tracker = SimpleTracker()

# ----------------- ROI -----------------
def get_center_trapezoid(frame_w, frame_h, w_scale=ROI_SCALE_W, h_scale=ROI_SCALE_H):
    top_w = int(frame_w * w_scale * 0.5)
    bottom_w = int(frame_w * w_scale)
    top_y = int(frame_h * (1.0 - h_scale))
    bottom_y = frame_h
    cx = frame_w // 2
    pts = np.array([
        (cx - top_w, top_y),
        (cx + top_w, top_y),
        (cx + bottom_w//2, bottom_y),
        (cx - bottom_w//2, bottom_y)
    ], dtype=np.int32)
    return pts

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

# ----------------- Gesture (MediaPipe Hands) -----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def detect_gesture_from_landmarks(landmarks):
    try:
        tips = [landmarks[i] for i in (4,8,12,16,20)]
        wrist = landmarks[0]
    except Exception:
        return None
    ys = [p.y for p in tips]
    if all(y < wrist.y - 0.03 for y in ys): return "Open Palm"
    if all(y > wrist.y + 0.02 for y in ys): return "Fist"
    if tips[0].y < wrist.y - 0.03 and tips[1].y > wrist.y + 0.01 and tips[2].y > wrist.y + 0.01: return "Thumbs Up"
    if tips[1].y < tips[2].y - 0.02 and tips[1].y < tips[0].y - 0.02: return "Pointing"
    return None

def landmarks_to_bbox_px(landmarks, w, h):
    xs = [int(l.x * w) for l in landmarks]
    ys = [int(l.y * h) for l in landmarks]
    return (max(min(xs),0), max(min(ys),0), min(max(xs),w-1), min(max(ys),h-1))

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    iw = max(0, min(ax2,bx2)-max(ax1,bx1)); ih = max(0, min(ay2,by2)-max(ay1,by1))
    inter = iw*ih
    if inter <= 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua>0 else 0.0

# ----------------- Face/smile -----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_expression(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces)==0: return None, None
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    (x,y,w,h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, SMILE_DETECT_MIN_NEIGHBORS)
    return ("Smiling" if len(smiles)>0 else "Neutral"), (x,y,w,h)

# ----------------- Emergency sound detection thread -----------------
audio_q = queue.Queue()

def sound_callback(indata, frames, time_info, status):
    if status: pass
    audio_q.put(indata.copy())

def sound_listener_loop():
    print("Audio thread started — listening for emergency sounds...")
    try:
        with sd.InputStream(channels=1, samplerate=AUDIO_SAMPLERATE, callback=sound_callback,
                            blocksize=int(AUDIO_SAMPLERATE*AUDIO_BLOCK_SECONDS)):
            while True:
                try:
                    block = audio_q.get(timeout=2.0)
                except queue.Empty:
                    continue
                data = np.squeeze(block)
                rms = float(np.sqrt(np.mean(data**2)))
                if rms > SOUND_RMS_THRESHOLD:
                    freqs = np.fft.rfftfreq(len(data), d=1.0/AUDIO_SAMPLERATE)
                    spectrum = np.abs(np.fft.rfft(data))
                    band_idx = np.where((freqs >= SIREN_FREQ_LOW) & (freqs <= SIREN_FREQ_HIGH))[0]
                    ratio = (np.sum(spectrum[band_idx]) / (np.sum(spectrum)+1e-9)) if len(band_idx)>0 else 0.0

                    # Two-tier speech
                    if ratio > SPECTRAL_ENERGY_RATIO_THRESH:
                        print(f"!!! DANGER siren/alarm (rms={rms:.3f}, ratio={ratio:.3f})")
                        speak("Danger alarm detected. Do not proceed.")
                    elif rms > SOUND_RMS_THRESHOLD * 3:
                        print(f"!!! Loud impact (rms={rms:.3f})")
                        speak("Loud noise detected. Be careful.")
    except Exception as e:
        print("Audio listener error:", e)
        print("Emergency sound detection disabled.")

# ----------------- Alert helper -----------------
def alert_action(obj_id, text, speak_text=None):
    global last_alerts
    with lock_alerts:
        now = time.time()
        last_time = last_alerts.get(obj_id, 0)
        if now - last_time < ALERT_COOLDOWN: return
        last_alerts[obj_id] = now
    if speak_text: speak(speak_text)
    print(f"[ALERT] {text} @ {time.strftime('%H:%M:%S')}")

# ----------------- MAIN LOOP -----------------
def run_camera_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam"); return

    fps_time = time.time(); frame_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    global last_gesture_time, gesture_last_spoken, last_direction_speech, last_direction_phrase

    # state for expression changes
    run_camera_loop.last_expr = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        roi_poly = get_center_trapezoid(w, h)

        # -------------- YOLO detection --------------
        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False, imgsz=640)[0]
        dets = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                x1,y1,x2,y2 = map(int, box.tolist())
                clsid = int(cls.tolist())
                label = COCO_NAMES[clsid] if clsid < len(COCO_NAMES) else str(clsid)
                if label not in ALERT_CLASSES: continue
                dets.append({'xyxy': (x1,y1,x2,y2), 'label': label, 'conf': float(conf.tolist())})
        now_ts = time.time()
        tracked = tracker.update(dets, now_ts)

        # make quick access map for person boxes (for gesture association)
        person_boxes = {obj['id']: obj['xyxy'] for obj in tracked if obj['label']=='person'}

        # -------------- draw ROI --------------
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_poly], (40,40,40))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [roi_poly], True, (0,255,255), 2)

        # sectors (for direction advice)
        left_band  = (0, 0, w//3, h)
        mid_band   = (w//3, 0, 2*w//3, h)
        right_band = (2*w//3, 0, w, h)

        def bbox_area(b):
            x1,y1,x2,y2=b; return max(0,(x2-x1))*max(0,(y2-y1))

        # -------------- gesture detection with person association --------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_res = hands_detector.process(frame_rgb)
        gesture_text = None
        if hands_res.multi_hand_landmarks:
            for hand_landmarks in hands_res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                g = detect_gesture_from_landmarks(hand_landmarks.landmark)
                if not g: continue
                gesture_text = g

                # associate this hand bbox to nearest/overlapping person
                hx1,hy1,hx2,hy2 = landmarks_to_bbox_px([l for l in hand_landmarks.landmark], w, h)
                best_pid, best_iou = None, 0.0
                for pid, pbox in person_boxes.items():
                    i = iou((hx1,hy1,hx2,hy2), pbox)
                    if i > best_iou:
                        best_iou, best_pid = i, pid

                # speak with cooldown or on-change
                say_str = f"Gesture detected: {g}" if best_pid is None else f"Person {best_pid} gesture: {g}"
                if time.time() - last_gesture_time > GESTURE_SPEECH_COOLDOWN or gesture_last_spoken != say_str:
                    gesture_last_spoken = say_str
                    last_gesture_time = time.time()
                    speak(say_str)

                # draw hand bbox + label
                cv2.rectangle(frame, (hx1,hy1), (hx2,hy2), (150,255,150), 2)
                cv2.putText(frame, (f"{g}" if best_pid is None else f"{g} (P{best_pid})"),
                            (hx1, max(15,hy1-6)), font, 0.6, (150,255,150), 2)

        # -------------- face / smile detection --------------
        expression, face_rect = detect_expression(frame)
        if expression:
            x,y,fw,fh = face_rect
            cv2.rectangle(frame, (x,y), (x+fw, y+fh), (255,200,0), 2)
            cv2.putText(frame, f"Expr: {expression}", (x, y-10), font, 0.6, (255,200,0), 2)
            if run_camera_loop.last_expr != expression:
                run_camera_loop.last_expr = expression
                speak(f"Person looks {expression}")

        # -------------- tracked objects (direction + approach) --------------
        left_blocked = False; mid_blocked = False; right_blocked = False
        for obj in tracked:
            x1,y1,x2,y2 = obj['xyxy']
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            label = obj['label']
            area_hist = obj['area_hist']
            cur_area = area_hist[-1]; prev_area = area_hist[-2] if len(area_hist)>1 else cur_area
            approaching = cur_area >= prev_area * APPROACH_AREA_GROWTH
            inside = point_in_poly((cx,cy), roi_poly)

            # sector for this object
            if cx < w//3: left_blocked  |= (bbox_area((x1,y1,x2,y2))>MIN_OBJECT_AREA_TO_BLOCK)
            elif cx > 2*w//3: right_blocked |= (bbox_area((x1,y1,x2,y2))>MIN_OBJECT_AREA_TO_BLOCK)
            else: mid_blocked   |= (bbox_area((x1,y1,x2,y2))>MIN_OBJECT_AREA_TO_BLOCK)

            # relative position text
            if cx < w//3: pos = "on the left"
            elif cx > 2*w//3: pos = "on the right"
            else: pos = "in front"

            color = (0,0,255) if inside else (200,200,200)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{label} {int(cur_area)}", (x1, y1-8), font, 0.6, color, 2)
            cv2.circle(frame, (cx,cy), 4, (0,255,0), -1)

            if inside:
                cv2.putText(frame, f"FRONT: {label.upper()}", (20,40), font, 1.0, (0,0,255), 3)
                msg = f"{label} approaching {pos}" if approaching else f"{label} {pos}"
                alert_action(obj['id'], msg, speak_text=msg)

        # -------------- Direction advisor (which way is clear?) --------------
        # Visual lane bars
        for (bx1,by1,bx2,by2), blocked, label in [
            (left_band, left_blocked, "LEFT"),
            (mid_band,  mid_blocked,  "MID"),
            (right_band,right_blocked,"RIGHT")
        ]:
            cv2.rectangle(frame, (bx1,0), (bx2,10), (0,0,255) if blocked else (0,200,0), -1)
            cv2.putText(frame, f"{label}:{'X' if blocked else 'OK'}", (bx1+10, 30), font, 0.6,
                        (0,0,255) if blocked else (0,200,0), 2)

        # Speech: prefer mid, else a free side; only talk if mid blocked or guidance changed
        now = time.time()
        phrase = None
        if mid_blocked:
            if not left_blocked and right_blocked:
                phrase = "Left side is clear. Go left."
            elif not right_blocked and left_blocked:
                phrase = "Right side is clear. Go right."
            elif not left_blocked and not right_blocked:
                phrase = "Left and right are clear. Choose a side."
            else:
                phrase = "All directions blocked. Please stop."
        else:
            # mid clear; occasionally remind only if previously said something else
            if last_direction_phrase not in (None, "Forward is clear. Go straight."):
                phrase = "Forward is clear. Go straight."

        if phrase and (now - last_direction_speech > DIRECTION_SPEECH_COOLDOWN or phrase != last_direction_phrase):
            last_direction_phrase = phrase
            last_direction_speech = now
            speak(phrase)

        # -------------- HUD --------------
        frame_count += 1
        fps = None
        if time.time() - fps_time > 1.0:
            fps = frame_count / (time.time() - fps_time)
            fps_time = time.time(); frame_count = 0
        if fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-140, 30), font, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Gesture: {gesture_text if gesture_text else '---'}", (10,70), font, 0.6, (200,200,50), 2)
        cv2.putText(frame, f"Expression: {expression if expression else '---'}", (10,100), font, 0.6, (200,200,50), 2)
        cv2.putText(frame,
            "Developed by Aluvala Ediga Harsha Vardhan Goud",
            (10, h-25),   # bottom-left position
            font,
            0.4,          # small size
            (180,180,180),
            1,
            cv2.LINE_AA)
        
    
        cv2.putText(frame, "Press q to quit", (10,h-10), font, 0.5, (240,240,240), 1)

        cv2.imshow("Al Assistive System for the Blind", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------- START -----------------
if __name__ == "__main__":
    print("Starting Al Assistive System for the Blind. Press q to quit.")
    t_audio = threading.Thread(target=sound_listener_loop, daemon=True)
    t_audio.start()
    run_camera_loop()
