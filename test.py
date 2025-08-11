import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_profile = pipeline.start(config)

device = pipeline_profile.get_device()
depth_sensor = device.first_depth_sensor()

# Short Range 설정 적용
depth_sensor.set_option(rs.option.laser_power, 84)       # 레이저 파워 낮춤
depth_sensor.set_option(rs.option.receiver_gain, 18)     # 수신기 게인 높임
depth_sensor.set_option(rs.option.min_distance, 190) # 최소 거리 190mm(0.19m)

print("Short Range 설정이 적용되었습니다.")

align = rs.align(rs.stream.color)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth = depth_frame.get_distance(x, y)
        if depth == 0:
            print(f"Clicked: ({x}, {y}) → invalid/no depth")
        else:
            dx, dy, dz = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
            print(f"2D:({x},{y}) → 3D:({dx:.3f},{dy:.3f},{dz:.3f})")

cv2.namedWindow('RealSense')
cv2.setMouseCallback('RealSense', mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
# 사용이 끝나면
pipeline.stop()
