# calib_collect_and_solve.py
# (1) S로 샘플 수집 (p_cam, p_base)
# (2) C로 강체변환 T_base_cam 계산/저장/출력
# (3) ESC로 종료
# 필요: pyrealsense2, opencv-python, mediapipe, ur_rtde

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time
import os
import csv
import math

# ====== 설정 ======
UR_IP = "127.0.0.1"       # URSim 도커를 로컬에서 돌리는 경우 그대로 OK
MODEL_PRESET = rs.l500_visual_preset.short_range  # L515 권장 프리셋
SAVE_DIR = "calib_out"
CAM_CSV = os.path.join(SAVE_DIR, "calib_cam_pts.csv")
BASE_CSV = os.path.join(SAVE_DIR, "calib_base_pts.csv")
T_FILE  = os.path.join(SAVE_DIR, "T_base_cam.npy")

# ====== 간단 필터(EMA) ======
class EMA3D:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.x = None
    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        if self.x is None:
            self.x = v.copy()
        else:
            self.x = self.alpha*v + (1-self.alpha)*self.x
        return self.x

# ====== Kabsch로 강체변환 추정 ======
def estimate_rigid_transform(cam_pts, base_pts):
    """
    cam_pts, base_pts: (N,3)
    반환: R(3x3), t(3,), T(4x4) s.t.  p_base = R @ p_cam + t
    """
    cam = np.asarray(cam_pts, float)
    base = np.asarray(base_pts, float)
    assert cam.shape == base.shape and cam.shape[1] == 3 and cam.shape[0] >= 3

    c_cam = cam.mean(axis=0)
    c_base = base.mean(axis=0)
    X = cam - c_cam
    Y = base - c_base

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_base - R @ c_cam

    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    # RMSE
    pred = (R @ cam.T).T + t
    rmse = np.sqrt(np.mean(np.sum((pred - base)**2, axis=1)))
    return R, t, T, rmse

# ====== UR RTDE ======
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

def get_ur_tcp(rtde_r):
    """UR 베이스 좌표계 TCP 포즈에서 위치만 (x,y,z) 리턴 (m)"""
    pose = rtde_r.getActualTCPPose()  # [x,y,z, rx, ry, rz]
    return np.array(pose[:3], dtype=float)

# ====== RealSense + MediaPipe 준비 ======
def setup_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipeline.start(cfg)

    device = prof.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, MODEL_PRESET)
    print("[RS] L515 preset:", MODEL_PRESET)

    depth_intrin = prof.get_stream(rs.stream.depth)\
        .as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)
    return pipeline, align, depth_intrin

def deproject(depth_intrin, px, py, depth_m):
    return np.array(
        rs.rs2_deproject_pixel_to_point(depth_intrin, [px, py], float(depth_m)),
        dtype=float
    )

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # UR 연결
    print("[UR] connecting to", UR_IP)
    rtde_c = RTDEControl(UR_IP)
    rtde_r = RTDEReceive(UR_IP)

    # RealSense/MP
    pipeline, align, depth_intrin = setup_realsense()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    ema = EMA3D(alpha=0.35)

    # 수집 버퍼
    cam_pts = []   # Nx3
    base_pts = []  # Nx3

    # 기존 파일을 발견하면 알려줌 (V로 병합 가능)
    existing = os.path.exists(CAM_CSV) and os.path.exists(BASE_CSV)
    if existing:
        print(f"[INFO] 기존 저장 파일 발견: {CAM_CSV}, {BASE_CSV}  (V키로 불러오기)")

    cv2.namedWindow("Calib")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            img = np.asanyarray(color.get_data())
            H, W, _ = img.shape

            # 손 검출
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            # UI 텍스트
            cv2.putText(img, "[S] sample  [C] solve  [U] undo  [V] load  [R] reset  [ESC] quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
            cv2.putText(img, f"samples: {len(cam_pts)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2)

            p_cam_smoothed = None

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                cx = int(lm.landmark[8].x * W)
                cy = int(lm.landmark[8].y * H)
                d = depth.get_distance(cx, cy)

                if d > 0:
                    p_cam = deproject(depth_intrin, cx, cy, d)  # [X,Y,Z] in meters
                    p_cam_smoothed = ema(p_cam)
                    cv2.circle(img, (cx, cy), 10, (0,255,255), -1)
                    cv2.putText(img, f"cam: {p_cam_smoothed[0]:.3f}, {p_cam_smoothed[1]:.3f}, {p_cam_smoothed[2]:.3f}",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,220,120), 2)
                else:
                    cv2.putText(img, "depth invalid", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                cv2.putText(img, "no hand", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("Calib", img)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key == ord('s'):  # Sample capture
                if p_cam_smoothed is None:
                    print("[WARN] 손/깊이 유효할 때만 캡처 가능")
                    continue
                p_base = get_ur_tcp(rtde_r)  # [x,y,z]
                cam_pts.append(p_cam_smoothed.copy())
                base_pts.append(p_base.copy())
                print(f"[CAPTURE] cam={p_cam_smoothed.round(4)}  base={p_base.round(4)}  (N={len(cam_pts)})")

            elif key == ord('u'):  # Undo
                if cam_pts:
                    cam_pts.pop()
                    base_pts.pop()
                    print("[UNDO] last sample removed. N=", len(cam_pts))

            elif key == ord('r'):  # Reset
                cam_pts.clear()
                base_pts.clear()
                print("[RESET] buffers cleared.")

            elif key == ord('v'):  # Load/merge existing CSVs
                if os.path.exists(CAM_CSV) and os.path.exists(BASE_CSV):
                    cam_old = np.loadtxt(CAM_CSV, delimiter=',')
                    base_old = np.loadtxt(BASE_CSV, delimiter=',')
                    if cam_old.ndim == 1: cam_old = cam_old[None, :]
                    if base_old.ndim == 1: base_old = base_old[None, :]
                    if cam_old.shape == base_old.shape and cam_old.shape[1] == 3:
                        cam_pts += [c for c in cam_old]
                        base_pts += [b for b in base_old]
                        print(f"[LOAD] merged {len(cam_old)} samples. N={len(cam_pts)}")
                    else:
                        print("[LOAD-ERR] CSV shape mismatch")
                else:
                    print("[LOAD] no csv to load.")

            elif key == ord('c'):  # Compute T
                if len(cam_pts) < 3:
                    print("[SOLVE] 최소 3점 필요. 현재:", len(cam_pts))
                    continue
                R, t, T, rmse = estimate_rigid_transform(np.array(cam_pts), np.array(base_pts))
                print("\n===== T_base_cam (p_base = R p_cam + t) =====")
                np.set_printoptions(suppress=True, precision=6)
                print(T)
                print(f"RMSE: {rmse*1000:.1f} mm")

                # 저장
                os.makedirs(SAVE_DIR, exist_ok=True)
                np.savetxt(CAM_CSV, np.array(cam_pts), delimiter=',')
                np.savetxt(BASE_CSV, np.array(base_pts), delimiter=',')
                np.save(T_FILE, T)
                print(f"[SAVE] {CAM_CSV}, {BASE_CSV}, {T_FILE}\n")
                cv2.putText(img, f"SOLVED! RMSE={rmse*1000:.1f}mm", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Calib", img)
                cv2.waitKey(500)

    finally:
        try: rtde_c.stopScript()
        except: pass
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
