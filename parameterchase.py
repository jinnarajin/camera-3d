
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# class KalmanFilter1D:
#     def __init__(self, process_variance, measurement_variance, estimated_error=1.0, initial_value=0.0):
#         self.process_variance = process_variance          # 프로세스 노이즈 분산
#         self.measurement_variance = measurement_variance  # 측정 노이즈 분산
#         self.estimated_error = estimated_error            # 추정 오차 공분산
#         self.posteri_estimate = initial_value             # 초기 상태 추정값

#     def update(self, measurement):
#         # 예측 단계
#         priori_estimate = self.posteri_estimate
#         priori_error = self.estimated_error + self.process_variance

#         # 칼만 이득 계산
#         kalman_gain = priori_error / (priori_error + self.measurement_variance)

#         # 상태 업데이트
#         self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)

#         # 오차 공분산 갱신
#         self.estimated_error = (1 - kalman_gain) * priori_error

#         return self.posteri_estimate


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_profile = pipeline.start(config)
align = rs.align(rs.stream.color)

device = pipeline_profile.get_device()
depth_sensor = device.first_depth_sensor()

depth_sensor.set_option(rs.option.laser_power, 84)       # 레이저 파워 낮춤
depth_sensor.set_option(rs.option.receiver_gain, 18)     # 수신기 게인 높임
depth_sensor.set_option(rs.option.min_distance, 250) # 최소 거리 250mm(0.25m)

print("Short Range 설정이 적용되었습니다.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils




# kf_x = KalmanFilter1D(process_variance=1e-5, measurement_variance=1e-2)
# kf_y = KalmanFilter1D(process_variance=1e-5, measurement_variance=1e-2)
# kf_z = KalmanFilter1D(process_variance=1e-3, measurement_variance=1e-1)

print("시스템 시작: 손가락 3D 좌표 실시간 추적 및 칼만 필터 안정화 적용 중...")

def get_stable_depth(depth_frame, x, y, window=3):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    total, count = 0, 0
    half = window // 2
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                d = depth_frame.get_distance(nx, ny)
                if d > 0:
                    total += d
                    count += 1
    return total / count if count else 0

try:
    while True:
        # 프레임 획득
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # 얼라인 처리로 깊이와 컬러 정렬

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = cv2.flip(color_image, 1) #기본이 좌우반전이라 

        img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # MediaPipe 기준 검지 손가락 끝은 landmark #8
                h, w, _ = color_image.shape
                cx = int(hand_landmarks.landmark[8].x * w)
                cy = int(hand_landmarks.landmark[8].y * h)
 
                # cx = min(max(cx, 0), w - 1)
                # cy = min(max(cy, 0), h - 1)

            # depth = get_stable_depth(depth_frame, cx, cy)
            depth = depth_frame.get_distance(cx, cy)  # 미터 단위 거리
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        
            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)  # [X, Y, Z]

            # stable_x = kf_x.update(point_3d[0])
            # stable_y = kf_y.update(point_3d[1])
            # stable_z = kf_z.update(point_3d[2])


            cv2.circle(color_image, (cx, cy), 10, (0, 255, 255), -1)
            cv2.putText(color_image, f"3D: X={point_3d[0]:.2f}m Y={point_3d[1]:.2f}m Z={point_3d[2]:.2f}m",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            # cv2.putText(color_image, f"3D: X={point_3d[0]:.2f}m Y={point_3d[1]:.2f}m Z={stable_z:.2f}m",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow('RealSense Color Image', color_image)
        if cv2.waitKey(1) == 27:  # ESC 누르면 종료
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    hands.close()