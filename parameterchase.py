
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# 파이프라인을 생성하고, 컬러/깊이 스트림 활성화
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
# pipeline.start(config)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

try:
    while True:
        # 프레임 획득
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # MediaPipe 기준 검지 손가락 끝은 landmark #8
                h, w, _ = color_image.shape
                cx = int(hand_landmarks.landmark[8].x * w)
                cy = int(hand_landmarks.landmark[8].y * h)

                
            depth = depth_frame.get_distance(cx, cy)  # 미터 단위 거리
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        
            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)  # [X, Y, Z]

            cv2.circle(color_image, (cx, cy), 10, (0, 255, 255), -1)
            cv2.putText(color_image, f"3D: X={point_3d[0]:.2f}m Y={point_3d[1]:.2f}m Z={point_3d[2]:.2f}m",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow('RealSense Color Image', color_image)
        if cv2.waitKey(1) == 27:  # ESC 누르면 종료
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    hands.close()