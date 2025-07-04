import cv2
import mediapipe as mp
import numpy as np
import os
from engagement_tracker import EngagementTracker

# --- Enhanced Face/Eye Tracking with Tilt Robustness and Improved Skin Color Estimation ---

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

def get_face_detector():
    face_proto = cv2.data.haarcascades.replace('haarcascades/', 'dnn/') + "deploy.prototxt"
    face_model = cv2.data.haarcascades.replace('haarcascades/', 'dnn/') + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    if os.path.isfile(face_proto) and os.path.isfile(face_model):
        net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
        return "dnn", net
    else:
        print("Warning: DNN face model files not found. Using Haar face detector.")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return "haar", face_cascade

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
nose_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_nose.xml'
nose_cascade = cv2.CascadeClassifier(nose_cascade_path) if os.path.isfile(nose_cascade_path) else None

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.10, beta=10)
    return enhanced

def get_gaze_direction(eye_roi):
    if eye_roi.size == 0:
        return "Unknown"
    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.equalizeHist(gray_eye)
    _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            w = eye_roi.shape[1]
            h = eye_roi.shape[0]
            horiz = "Center"
            vert = "Center"
            if cx < w // 3:
                horiz = "Left"
            elif cx > 2 * w // 3:
                horiz = "Right"
            if cy < h // 3:
                vert = "Up"
            elif cy > 2 * h // 3:
                vert = "Down"
            if horiz == "Center" and vert == "Center":
                return "Looking Center"
            elif vert != "Center":
                return f"Looking {vert}"
            else:
                return f"Looking {horiz}"
    return "Unknown"

def draw_engagement_meter(frame, engagement, x=None, y=None, w=400, h=24):
    if x is None:
        x = frame.shape[1]//2 - w//2
    if y is None:
        y = 20
    bar_bg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        ratio = i / w
        if ratio < 0.5:
            color = (0, 180, 255)
        else:
            color = (0, 255, 0)
        bar_bg[:, i] = color
    fill = int(w * min(max(engagement, 0), 1))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, :fill] = 255
    bar_fg = cv2.bitwise_and(bar_bg, bar_bg, mask=mask)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    roi = frame[y:y+h, x:x+w]
    cv2.addWeighted(bar_fg, 0.8, roi, 0.2, 0, roi)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 220, 220), 2)
    text = f"Engagement: {int(engagement*100)}%"
    cv2.putText(frame, text, (x+10, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def style_text(frame, text, org, font_scale=0.7, color=(255,255,255), thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def detect_faces_dnn(frame, net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            bw, bh = x2-x1, y2-y1
            if bw > 40 and bh > 40:
                faces.append((x1, y1, bw, bh))
    return faces

def detect_faces_haar(gray, face_cascade):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return faces

def estimate_skin_color(face_roi):
    # Improved: Use both YCrCb and HSV for better robustness, especially for light/white skin
    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    y, cr, cb = cv2.split(ycrcb)
    # Mask for skin pixels (broad range)
    skin_mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_mask_hsv = cv2.inRange(hsv, (0, 10, 60), (20, 150, 255))
    skin_mask = cv2.bitwise_or(skin_mask_ycrcb, skin_mask_hsv)
    skin_pixels = face_roi[skin_mask > 0]
    if skin_pixels.size == 0:
        # fallback: use all pixels
        skin_pixels = face_roi.reshape(-1, 3)
    mean_bgr = np.mean(skin_pixels, axis=0)
    mean_b, mean_g, mean_r = mean_bgr
    mean_brightness = np.mean(cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2GRAY))
    # Fix: Invert the logic for black/white detection
    if mean_brightness > 200:
        return "Black"
    elif mean_brightness < 80:
        return "White"
    else:
        return "Normal"

def rotate_image(image, angle):
    # Rotate image around its center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def detect_eyes_with_tilt(roi_gray, roi_color, max_angle=30, step=10):
    # Try to detect eyes at different tilt angles
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.08, minNeighbors=3, minSize=(16, 16))
    if len(eyes) >= 2:
        return eyes, 0  # Found eyes, no tilt needed
    # Try rotating both directions
    for angle in range(-max_angle, max_angle+1, step):
        if angle == 0:
            continue
        rotated_gray = rotate_image(roi_gray, angle)
        rotated_color = rotate_image(roi_color, angle)
        eyes = eye_cascade.detectMultiScale(rotated_gray, scaleFactor=1.08, minNeighbors=3, minSize=(16, 16))
        if len(eyes) >= 2:
            return eyes, angle
    return [], 0

def draw_face_mesh(frame, landmarks, color=(0,255,0), radius=1, thickness=1):
    # Draw small circles at each landmark point
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    # Draw dotted lines between key facial regions (jaw, eyebrows, nose, eyes, mouth)
    mesh_lines = [
        list(range(0, 17)),      # Jaw
        list(range(17, 22)),     # Right eyebrow
        list(range(22, 27)),     # Left eyebrow
        list(range(27, 31)),     # Nose bridge
        list(range(31, 36)),     # Lower nose
        list(range(36, 42)),     # Right eye
        list(range(42, 48)),     # Left eye
        list(range(48, 60)),     # Outer lip
        list(range(60, 68)),     # Inner lip
    ]
    for group in mesh_lines:
        for i in range(len(group)-1):
            pt1 = landmarks[group[i]]
            pt2 = landmarks[group[i+1]]
            # Draw dotted line
            for alpha in np.linspace(0, 1, 10):
                x = int(pt1[0] * (1-alpha) + pt2[0] * alpha)
                y = int(pt1[1] * (1-alpha) + pt2[1] * alpha)
                cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

def main():
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        print(f"Running on video: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        print("Running on webcam.")
        video_path = "webcam"

    # Initialize EngagementTracker
    tracker = EngagementTracker()
    tracker.set_video_name(video_path)

    window_name = 'Face Engagement Analyzer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 600)

    engagement_level = 0.0
    decay_rate = 0.02

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video or failed to capture frame.")
            break

        frame = cv2.resize(frame, (900, 600))
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        engagement = 0.0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    display, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
                )
                engagement = 1.0
                break

        # Engagement smoothing
        if engagement > engagement_level:
            engagement_level = engagement
        else:
            engagement_level = max(0.0, engagement_level - decay_rate)

        # Add engagement score to tracker
        tracker.add_score(engagement_level)

        # Draw engagement meter
        draw_engagement_meter(display, engagement_level)

        cv2.imshow(window_name, enhance_contrast(display))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save engagement data to Excel
    tracker.save_to_excel()

if __name__ == "__main__":
    main()