
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time
import csv
import os
import datetime

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_profile = pipeline.start(config)
align = rs.align(rs.stream.color)

device = pipeline_profile.get_device()
depth_sensor = device.first_depth_sensor()

# depth_sensor.set_option(rs.option.laser_power, 84)       # 레이저 파워 낮춤
# depth_sensor.set_option(rs.option.receiver_gain, 18)     # 수신기 게인 높임
# depth_sensor.set_option(rs.option.min_distance, 0) # 최소 거리 250mm(0.25m) 수동
depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range) #자동

print("Short Range 설정이 적용되었습니다.")
depth_intrin = pipeline_profile.get_stream(rs.stream.depth)\
    .as_video_stream_profile().get_intrinsics()



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

print("손가락도 락이다.")

# ==== 상태 변수 ====
started = False
ref_point = None  # (X0, Y0, Z0)
logging_enabled = False   # REC 토글
detect_paused   = False   # 인식 일시정지 토글(옵션)
pending_start = False
segment_id = 0

# === ADDED: 로그 파일 핸들 ===
log_file = None           # === ADDED ===
log_writer = None         # === ADDED ===

# ==== 버튼 영역 설정(좌상단 10,10 에 120x40 박스) ====
BTN_X, BTN_Y, BTN_W, BTN_H = 10, 10, 120, 40

# === ADDED: 로그 파일 오픈/클로즈 함수 ===
def open_new_log():
    """REC 시작 시 새 CSV를 logs/ 폴더에 생성"""
    global log_file, log_writer, segment_id
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    fname = f"finger_points_{ts}_seg{segment_id}.csv"
    path = os.path.join("logs", fname)
    log_file = open(path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['ts', 'px', 'py', 'depth_m',
                         'X', 'Y', 'Z', 'relX', 'relY', 'relZ', 'started'])
    print("logging to:", path)

def close_log():
    """REC 정지 시 CSV 닫기"""
    global log_file, log_writer
    try:
        if log_file is not None:
            log_file.close()
    except Exception:
        pass
    log_file = None
    log_writer = None



# 마우스 콜백: 버튼 클릭 처리
def on_mouse(event, x, y, flags, param):
    # ✅ 추가: 전역 상태로 선언
    global started, ref_point, last_XYZ, logging_enabled, segment_id, pending_start  # <<< ADDED

    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_X <= x <= BTN_X+BTN_W and BTN_Y <= y <= BTN_Y+BTN_H:
            if not started:
                # 손 좌표가 이미 있으면 즉시 시작 + REC ON
                if last_XYZ is not None:
                    ref_point = last_XYZ  # (X0, Y0, Z0)
                    started = True
                    if not logging_enabled:
                        logging_enabled = True
                        segment_id += 1
                        open_new_log()
                else:
                    # ✅ 손이 아직 없으면 다음 유효 좌표 때 자동 시작
                    pending_start = True  # <<< ADDED
            else:
                ref_point = None
                started = False
                if logging_enabled:
                    logging_enabled = False
                    close_log()


# 윈도우/마우스 콜백 등록 (루프 시작 전에 1회)
cv2.namedWindow('RealSense Color Image')
cv2.setMouseCallback('RealSense Color Image', on_mouse)

# # ==== (선택) CSV 로깅 준비 ====
# log_path = 'finger_points.csv'
# log_file = open(log_path, 'w', newline='', encoding='utf-8')
# log_writer = csv.writer(log_file)
# log_writer.writerow(['ts', 'px', 'py', 'depth_m',
#                      'X', 'Y', 'Z', 'relX', 'relY', 'relZ', 'started'])

# 루프 안에서 매 프레임 갱신할 임시 보관
last_XYZ = None



try:
    while True:
        # 프레임 획득
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        # depth_image = np.asanyarray(depth_frame.get_data())

        #그리드 회색
        h, w, _ = color_image.shape
        grid_rows = 4  
        grid_cols = 4 
        grid_color = (128, 128, 128)  

        for i in range(1, grid_rows):
            y = int(h * i / grid_rows)
            cv2.line(color_image, (0, y), (w, y), grid_color, 1)

        for j in range(1, grid_cols):
            x = int(w * j / grid_cols)
            cv2.line(color_image, (x, 0), (x, h), grid_color, 1)

        # --- 버튼 UI 그리기 ---
        btn_label = "START" if not started else "RESET (R)"
        cv2.rectangle(color_image, (BTN_X, BTN_Y), (BTN_X+BTN_W, BTN_Y+BTN_H), (80,80,80), -1)
        cv2.putText(color_image, btn_label, (BTN_X+8, BTN_Y+27),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

         # === ADDED: REC 상태 점등 표시 ===
        rec_color = (0,0,255) if logging_enabled else (80,80,80)
        cv2.circle(color_image, (BTN_X+BTN_W+20, BTN_Y+20), 8, rec_color, -1)
        cv2.putText(color_image, "REC" if logging_enabled else "IDLE",
                    (BTN_X+BTN_W+35, BTN_Y+27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        overlay_lines = []  # 화면에 쓸 텍스트 라인

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # MediaPipe 기준 검지 손가락 끝은 landmark #8
                h, w, _ = color_image.shape
                cx = int(hand_landmarks.landmark[8].x * w)
                cy = int(hand_landmarks.landmark[8].y * h)
 

            depth = depth_frame.get_distance(cx, cy)

            if depth > 0:
                
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)  # [X, Y, Z]
                X, Y, Z = point_3d 
                dispX, dispY = 1*X, -1*Y
                last_XYZ=(X,Y,Z)
                # ✅ pending_start가 켜져 있으면, 이제 유효 좌표가 생겼으므로 여기서 기준점+REC ON
                if pending_start:
                    ref_point = (X, Y, Z)
                    started = True
                    if not logging_enabled:
                        logging_enabled = True
                        segment_id += 1
                        open_new_log()
                    pending_start = False

                #상대 좌표    
                if started and ref_point is not None:
                        relX = X - ref_point[0]
                        relY = Y - ref_point[1]
                        relZ = Z - ref_point[2]
                else:
                        relX = relY = relZ = 0.0

                if depth <0.10:  # 자동적으로 18이하면 없어지긴한데 오류를 예방하기 위해
                        status_text = "Too Close!"
                        color = (0, 0, 255)  # 빨강
                else:
                        status_text = "OK"
                        color = (0, 255, 0)  # 초록

                #그리드 셀 실제 크기 계산
                cell_width_cm = (w / grid_cols) * depth / depth_intrin.fx * 100
                cell_height_cm = (h / grid_rows) * depth / depth_intrin.fy * 100

                cv2.circle(color_image, (cx, cy), 10, (0, 255, 255), -1)

                overlay_lines.append(f"{status_text} ABS: X={dispX:.3f} Y={dispY:.3f} Z={Z:.3f} m")
                if started:
                    overlay_lines.append(f"REL: dX={relX:.3f} dY={relY:.3f} dZ={relZ:.3f} m")
                overlay_lines.append(f"Cell: {cell_width_cm:.1f} x {cell_height_cm:.1f} cm")

                # === ADDED: REC 켠 동안에만 CSV 기록 ===
                if logging_enabled and (log_writer is not None):
                    ts = time.time()
                    log_writer.writerow([ts, cx, cy, depth, X, Y, Z, relX, relY, relZ, int(started)])

            else:
            # depth==0인 경우(유효하지 않음)
                pass
        
        if pending_start and not started:
            cv2.putText(color_image, "Waiting for hand to start...", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            
        y0 = 120 if (pending_start and not started) else 65
        for i, line in enumerate(overlay_lines[:3]):
            cv2.putText(color_image, line, (10, y0 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

        # y0 = 65  # 버튼 아래부터 출력
        # for i, line in enumerate(overlay_lines[:3]):  # 3줄만 표시
        #         cv2.putText(color_image, line, (10, y0 + i*22),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)


            # cv2.putText(color_image, f"{status_text} 3D: X={dispX:.2f}m Y={dispY:.2f}m Z={Z:.2f}m",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # cv2.putText(color_image,f"Cell Size: {cell_width_cm:.1f}cm x {cell_height_cm:.1f}cm",(10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


        cv2.imshow('RealSense Color Image', color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 누르면 종료
            if logging_enabled:
                close_log()
            break
        elif key == ord('s'):  # START: 현재 좌표를 기준점으로 + 자동 REC ON
            if last_XYZ is not None:
                ref_point = last_XYZ
                started = True
                if not logging_enabled:
                    logging_enabled = True
                    segment_id += 1
                    open_new_log()
            else:
                pending_start = True  # ✅ 손이 아직 없으면 다음 유효 좌표 때 자동 시작      
        elif key == ord('r'):  # RESET
            ref_point = None
            started = False
            if logging_enabled:     # ✅ REC가 켜져있으면
                logging_enabled = False
                close_log() 
        elif key == ord('l'):  # === ADDED: REC 토글 ===
            logging_enabled = not logging_enabled
            if logging_enabled:
                segment_id += 1
                open_new_log()
            else:
                close_log()


finally:
    try:
        close_log()
    except Exception:
        pass
    pipeline.stop()
    cv2.destroyAllWindows()
    hands.close()