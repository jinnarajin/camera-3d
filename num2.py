
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time
import csv
import os
import datetime
# ======================= 2단계: 후처리/변환 유틸 =========================
from collections import deque
import math


class OutlierGate:
    """이상치 게이트: 깊이=0, 점프(속도) 과도, 범위 밖 제거"""
    def __init__(self, v_max=2.0, bounds=None):
        # v_max: m/s, 프레임 간 속도 제한(상황에 맞게 0.8~2.0 권장)
        self.v_max = v_max
        self.prev = None
        self.prev_t = None
        self.bounds = bounds  # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or None

    def accept(self, t, p):
        if p is None: 
            return False
        if self.bounds is not None:
            (xmin,xmax),(ymin,ymax),(zmin,zmax) = self.bounds
            if not (xmin<=p[0]<=xmax and ymin<=p[1]<=ymax and zmin<=p[2]<=zmax):
                return False
        if self.prev is not None and self.prev_t is not None:
            dt = max(1e-6, t - self.prev_t)
            v = np.linalg.norm(p - self.prev) / dt
            if v > self.v_max:   # 비현실적인 급변 → 버림
                return False
        self.prev, self.prev_t = p, t
        return True

class OneEuroFilter:
    """One Euro Filter: 지터는 줄이고 반응성은 유지 (VR/손추적에서 표준)"""
    def __init__(self, freq=60.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        # freq: 입력 샘플링 추정(Hz), min_cutoff: 기본 컷오프, beta: 속도비례 가변성, d_cutoff: 도함수 필터 컷오프
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x
        dt = max(1e-6, t - self.t_prev)
        self.t_prev = t

        # 1) 속도 추정에 대한 저역통과
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # 2) 가변 컷오프로 위치 필터
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

class EMA3D:
    """지수이동평균(간단·안정). alpha=0.2~0.5 권장"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.state = None
    def __call__(self, x):
        if self.state is None:
            self.state = x.copy()
        else:
            self.state = self.alpha * x + (1 - self.alpha) * self.state
        return self.state

class VelocityLimiter:
    """속도 제한으로 스파이크 추가 완화"""
    def __init__(self, v_max=0.8):  # m/s
        self.v_max = v_max
        self.prev = None
        self.prev_t = None
    def __call__(self, t, x):
        if self.prev is None:
            self.prev = x.copy()
            self.prev_t = t
            return x
        dt = max(1e-6, t - self.prev_t)
        step = x - self.prev
        step_max = self.v_max * dt
        if np.linalg.norm(step) > step_max:
            step = step * (step_max / (np.linalg.norm(step) + 1e-9))
        self.prev = self.prev + step
        self.prev_t = t
        return self.prev

# === 카메라 -> 로봇 베이스 좌표 변환 준비 (캘리브레이션 결과행렬) ===
# 실제 캘리브레이션 후 값으로 바꿔 넣으세요.
T_base_cam = np.eye(4)  # placeholder (4x4)
# 예) R|t 구성:
# R = np.eye(3); t = np.array([0.4, -0.1, 0.25])
# T_base_cam[:3,:3] = R; T_base_cam[:3,3] = t

def cam_to_base(p_cam):
    """p_cam: (3,), meters -> p_base: (3,)"""
    p = np.ones(4); p[:3] = p_cam
    pb = T_base_cam @ p
    return pb[:3]

def pose_to_ur_tcp(p_base, fixed_rpy=(0, 3.1416, 0)):
    """UR servo/speedL용 TCP 포맷 [x,y,z,rx,ry,rz] (회전은 고정 RPY->axis-angle 간단 근사)
       여기선 간단히 RPY를 axis-angle로 변환하지 않고 rx,ry,rz에 RPY를 직접 넣어두는 placeholder."""
    x, y, z = p_base
    rx, ry, rz = fixed_rpy  # 실제로는 RPY->axis-angle 변환 필요
    return [x, y, z, rx, ry, rz]
# ======================================================================

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_profile = pipeline.start(config)
align = rs.align(rs.stream.color)

device = pipeline_profile.get_device()
depth_sensor = device.first_depth_sensor()

depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range) #자동

print("Short Range 설정이 적용되었습니다.")
depth_intrin = pipeline_profile.get_stream(rs.stream.depth)\
    .as_video_stream_profile().get_intrinsics()



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

print("손가락도 락이다.")

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
                    pending_start = True  
            else:
                ref_point = None
                started = False
                if logging_enabled:
                    logging_enabled = False
                    close_log()


# 윈도우/마우스 콜백 등록 (루프 시작 전에 1회)
cv2.namedWindow('RealSense Color Image')
cv2.setMouseCallback('RealSense Color Image', on_mouse)

last_XYZ = None

# === 루프 들어가기 전에 1회 ===
gate      = OutlierGate(v_max=1.2, bounds=None)     # v_max, bounds 튜닝
one_euro  = OneEuroFilter(freq=30.0, min_cutoff=1.2, beta=0.02, d_cutoff=1.0)
ema       = EMA3D(alpha=0.35)
vel_limit = VelocityLimiter(v_max=0.6)               # 로봇이 따라가기 쉬운 속도로

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
                # ======== (루프 내부) 2단계 처리 시작 ========
                t_now = time.time()
                raw =  np.array([X, Y, Z])
                # 0) 이상치 게이트 (깊이 0/과속/범위 밖 제거)
                if not gate.accept(t_now,raw):
                    # 게이트 통과 못하면 표시만 하고 기록/로봇전달 스킵
                    pass
                else:
                    raw = np.array([X, Y, Z])

                    # 1) 1유로 필터 또는 EMA 선택 (하나만 써도 됨, 둘 다 써도 OK)
                    sm1 = one_euro(t_now, raw)      # 반응성 필요시
                    sm2 = ema(sm1)                  # 잔여 지터 추가 완화

                    # 2) 속도 제한(안정성 강화)
                    smoothed = vel_limit(t_now, sm2)

                    # 3) (선택) 상대좌표 적용
                    if started and ref_point is not None:
                        rel = smoothed - np.array(ref_point)
                    else:
                        rel = np.zeros(3)

                    # 4) 로봇 베이스 좌표로 변환 (캘리브레이션 완료 후 활성화)
                    p_base = cam_to_base(smoothed)  # TODO: T_base_cam 실제 값으로 교체

                    # 5) UR 명령 포맷으로 포장 (연동은 3단계에서)
                    ur_tcp = pose_to_ur_tcp(p_base)
                # ======== (루프 내부) 2단계 처리 끝 ========
                # 속도 계산 (m/s)
                    if 'prev_pos' not in locals():
                        prev_pos = smoothed
                        prev_time = t_now
                    else:
                        dt = t_now - prev_time
                        if dt > 0:
                            velocity = np.linalg.norm(smoothed - prev_pos) / dt
                            prev_pos = smoothed
                            prev_time = t_now

                            # 현장 체크 표시
                            if velocity < 0.02:
                                speed_status = "STOP"
                                color = (0, 255, 0)
                            elif velocity < 0.2:
                                speed_status = "SLOW"
                                color = (255, 255, 0)
                            else:
                                speed_status = "FAST"
                                color = (0, 0, 255)

                            # 화면 출력
                            cv2.putText(color_image, f"{speed_status} ({velocity:.2f} m/s)",
                                        (10, y0 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


                dispX, dispY = 1*X, -1*Y
                last_XYZ=(X,Y,Z)
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