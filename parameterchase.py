
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

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



        img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # MediaPipe 기준 검지 손가락 끝은 landmark #8
                h, w, _ = color_image.shape
                cx = int(hand_landmarks.landmark[8].x * w)
                cy = int(hand_landmarks.landmark[8].y * h)
 

            depth = depth_frame.get_distance(cx, cy)
            if depth <= 0:
                continue
            print(depth)
            
            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)  # [X, Y, Z]


            
            X, Y, Z = point_3d 
            dispX, dispY = 1*X, -1*Y
            if depth <0.10:  # 자동적으로 18이하면 없어지긴한데 오류를 예방하기 위해
                    status_text = "Too Close!"
                    color = (0, 0, 255)  # 빨강
            else:
                    status_text = "OK"
                    color = (0, 255, 0)  # 초록

            # 한 칸 실제 크기 계산 (cm)
            cell_width_cm = (w / grid_cols) * depth / depth_intrin.fx * 100
            cell_height_cm = (h / grid_rows) * depth / depth_intrin.fy * 100

            cv2.circle(color_image, (cx, cy), 10, (0, 255, 255), -1)

            cv2.putText(color_image, f"{status_text} 3D: X={dispX:.2f}m Y={dispY:.2f}m Z={Z:.2f}m",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(color_image,f"Cell Size: {cell_width_cm:.1f}cm x {cell_height_cm:.1f}cm",(10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


        cv2.imshow('RealSense Color Image', color_image)
        if cv2.waitKey(1) == 27:  # ESC 누르면 종료
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    hands.close()