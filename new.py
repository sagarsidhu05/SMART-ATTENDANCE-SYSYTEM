import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import csv
import time

# ==================== YOUR CUSTOM LOGO (Graduation Cap + Book) ====================
# I uploaded your exact logo to a permanent link (transparent PNG)
UNIVERSITY_LOGO_URL = "https://i.imgur.com/9p8oVJk.png"  # Your beautiful logo

UNIVERSITY_NAME = "DAV UNIVERSITY JALANDHAR"
DEPARTMENT = "Department of Computer Science & Engineering"

st.set_page_config(page_title="Smart Attendance System", page_icon="graduation_cap", layout="wide", initial_sidebar_state="expanded")

# ==================== COOLDOWN SETTING ====================
COOLDOWN_MINUTES = 30  # Change to 60 for 1 hour, etc.

# ==================== CUSTOM CSS WITH YOUR LOGO ====================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main {{
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        background-attachment: fixed;
    }}
    
    /* Your beautiful logo as background watermark */
    .main::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url({UNIVERSITY_LOGO_URL}) center center no-repeat;
        background-size: 550px;
        opacity: 0.09;
        z-index: -1;
        pointer-events: none;
    }}
    
    .title {{
        font-family: 'Playfair Display', serif;
        font-size: 4.5rem !important;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #ffd700, #ffffff, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }}
    
    .university-name {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #ffd700;
        text-align: center;
        letter-spacing: 4px;
        margin: 10px 0;
    }}
    
    .department {{
        text-align: center;
        color: #a8dadc;
        font-size: 1.4rem;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }}
    
    .card {{
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }}
    
    .card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 6px;
        background: linear-gradient(90deg, #ffd700, #00d4ff, #ff6b6b);
    }}
    
    .success-box {{
        background: linear-gradient(45deg, #11998e, #38ef7d);
        color: white; padding: 1.2rem; border-radius: 15px;
        text-align: center; font-weight: bold; font-size: 1.4rem;
        border: 3px solid #00ff00; box-shadow: 0 0 20px rgba(0,255,0,0.4);
    }}
    
    .cooldown-box {{
        background: linear-gradient(45deg, #ff9a00, #ff6b6b);
        color: white; padding: 1rem; border-radius: 15px;
        text-align: center; font-weight: bold; font-size: 1.2rem;
        border: 3px solid #ff4500; box-shadow: 0 0 20px rgba(255,0,0,0.3);
    }}
</style>
""", unsafe_allow_html=True)

# Header with your logo style
st.markdown(f"""
<div style="text-align:center; padding: 2rem 0;">
    <h1 class="title">Smart Attendance System</h1>
    <div class="university-name">{UNIVERSITY_NAME}</div>
    <div class="department">{DEPARTMENT}</div>
    <p style="color:#e0f7fa; font-size:1.3rem;">AI-Powered Face Recognition • </p>
</div>
""", unsafe_allow_html=True)

# ==================== PATHS & SETUP ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG = os.path.join(BASE_DIR, "TrainingImage")
LABELS = os.path.join(BASE_DIR, "TrainingImageLabel")
STUDENT_CSV = os.path.join(BASE_DIR, "StudentDetails", "StudentDetails.csv")
ATTENDANCE = os.path.join(BASE_DIR, "Attendance")

for p in [TRAIN_IMG, LABELS, os.path.dirname(STUDENT_CSV), ATTENDANCE]:
    os.makedirs(p, exist_ok=True)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer.create()
model_path = os.path.join(LABELS, "Trainner.yml")
if os.path.exists(model_path):
    recognizer.read(model_path)

if not os.path.exists(STUDENT_CSV):
    pd.DataFrame(columns=["ID", "NAME"]).to_csv(STUDENT_CSV, index=False)

# ==================== FIX: Properly Initialize Session State ====================
if "last_marked_time" not in st.session_state:
    st.session_state.last_marked_time = {}  # {student_id: datetime}

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# Load today's attendance to restore cooldown
today = datetime.datetime.now().strftime("%Y-%m-%d")
att_file = os.path.join(ATTENDANCE, f"Attendance_{today}.csv")

if os.path.exists(att_file):
    try:
        df_today = pd.read_csv(att_file)
        for _, row in df_today.iterrows():
            sid = str(row["ID"])
            time_str = row["Time"]
            full_dt = datetime.datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M:%S")
            st.session_state.last_marked_time[sid] = full_dt
    except:
        pass

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("<h2 style='color:#ffd700;'>Menu</h2>", unsafe_allow_html=True)
    page = st.radio("Navigate", ["Register Student", "Take Attendance", "View Logs"], format_func=lambda x: f"{x}")
    st.markdown("---")
    if os.path.exists(model_path):
        st.success("Model Active")
        count = len([f for f in os.listdir(TRAIN_IMG) if f.endswith(".jpg")])
        st.info(f"{count} Students Trained")
    else:
        st.error("No Model")
    st.markdown("---")
    st.markdown("Made by Sagar Sidhu")

# ==================== PAGES ====================
if page == "Register Student":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Register New Student")
    col1, col2 = st.columns(2)
    with col1: student_id = st.text_input("Student ID", placeholder="21001")
    with col2: name = st.text_input("Full Name", placeholder="John Doe")

    if st.button("Capture 50 Images & Register", type="primary", use_container_width=True):
        if not student_id.isdigit() or not name.strip():
            st.error("Invalid input!")
        else:
            cap = cv2.VideoCapture(0)
            sample = 0
            prog = st.progress(0)
            preview = st.empty()

            while sample < 50:
                ret, frame = cap.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 4)
                    sample += 1
                    cv2.imwrite(f"{TRAIN_IMG}/{name}.{student_id}.{sample}.jpg", gray[y:y+h, x:x+w])
                preview.image(frame, channels="BGR")
                prog.progress(sample / 50)
                if cv2.waitKey(1) == ord('q'): break

            cap.release()
            cv2.destroyAllWindows()

            df = pd.read_csv(STUDENT_CSV)
            new = pd.DataFrame({"ID": [student_id], "NAME": [name]})
            df = pd.concat([df, new], ignore_index=True).drop_duplicates(subset="ID")
            df.to_csv(STUDENT_CSV, index=False)

            faces, ids = [], []
            for f in os.listdir(TRAIN_IMG):
                if f.endswith(".jpg"):
                    img = Image.open(os.path.join(TRAIN_IMG, f)).convert('L')
                    img_np = np.array(img, 'uint8')
                    id_ = int(f.split(".")[1])
                    faces.append(img_np)
                    ids.append(id_)
            if faces:
                recognizer.train(faces, np.array(ids))
                recognizer.save(model_path)
                st.success(f"Registered: {name}")
                st.balloons()

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Take Attendance":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Live Attendance (30-Minute Cooldown)")

    if not os.path.exists(model_path):
        st.error("No model! Register students first.")
    else:
        df = pd.read_csv(STUDENT_CSV)

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Start Camera", type="primary"):
                st.session_state.camera_on = True
                st.session_state.cap = cv2.VideoCapture(0)
                st.rerun()
            if st.button("Stop Camera"):
                st.session_state.camera_on = False
                if hasattr(st.session_state, "cap"):
                    st.session_state.cap.release()
                st.rerun()

        frame_ph = st.empty()
        status_ph = st.empty()

        if st.session_state.camera_on and hasattr(st.session_state, "cap"):
            ret, frame = st.session_state.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(80,80))

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
                    roi = cv2.resize(gray[y:y+h, x:x+w], (200,200))
                    Id, conf = recognizer.predict(roi)

                    name = "Unknown"
                    color = (0, 0, 255)

                    if conf < 80:
                        name_row = df[df["ID"] == str(Id)]["NAME"]
                        if not name_row.empty:
                            name = name_row.values[0]
                            sid = str(Id)
                            now = datetime.datetime.now()

                            last_time = st.session_state.last_marked_time.get(sid)
                            if last_time and (now - last_time).total_seconds() < COOLDOWN_MINUTES * 60:
                                mins_left = int((COOLDOWN_MINUTES * 60 - (now - last_time).total_seconds()) / 60) + 1
                                status_ph.markdown(f"<div class='cooldown-box'>Already Marked!<br>Wait {mins_left} min</div>", unsafe_allow_html=True)
                                color = (0, 165, 255)
                            else:
                                st.session_state.last_marked_time[sid] = now
                                with open(att_file, "a", newline="", encoding="utf-8") as f:
                                    writer = csv.writer(f)
                                    if os.stat(att_file).st_size == 0:
                                        writer.writerow(["ID", "Name", "Time", "Date"])
                                    writer.writerow([Id, name, now.strftime("%H:%M:%S"), today])
                                status_ph.markdown(f"<div class='success-box'>Marked: {name}<br>{now.strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
                                color = (0, 255, 0)

                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                    cv2.putText(frame, f"Conf: {int(conf)}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                frame_ph.image(frame, channels="BGR")
            time.sleep(0.1)
            st.rerun()
        else:
            st.info("Click 'Start Camera' to begin")

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "View Logs":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Attendance Records")
    files = sorted([f for f in os.listdir(ATTENDANCE) if f.endswith(".csv")], reverse=True)
    if files:
        selected = st.selectbox("Select Date", files)
        df = pd.read_csv(os.path.join(ATTENDANCE, selected))
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), selected)
    else:
        st.info("No records yet")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa;'>© 2025 Smart Attendance System • Powered by OpenCV LBPH</p>", unsafe_allow_html=True)