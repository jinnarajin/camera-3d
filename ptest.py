# import torch
# print("GPU 사용 가능 여부:", torch.cuda.is_available())
# print("사용 가능한 GPU 수:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         print(f"GPU {i} 이름:", torch.cuda.get_device_name(i))

from ultralytics import YOLO
import cv2

# 공식 pretrained 손 키포인트 모델 로드
model = YOLO("yolo11n-pose.pt")
ipath= "image/i1.png"

# 이미지 또는 비디오 경로 지정하여 추론 수행 (GPU가 없으면 device="cpu"로 설정)
results = model.predict(source=ipath, device="cuda")  # GPU 환경일 경우
# results = model.predict(source="path/to/sample_image.jpg", device="cpu")  # CPU 환경일 경우
img = cv2.imread(ipath)

index_fingertip_idx = 8 


for result in results:
    result.show()
# 추론 결과 시각화
for result in results:
    keypoints = result.keypoints  # shape: (num_keypoints, 3) - x, y, confidence

    if keypoints is not None:
        # 검지손가락 끝 좌표 추출
        fingertip = keypoints[0][index_fingertip_idx]  # 첫 번째 손, 검지 끝
        x, y = int(fingertip[0]), int(fingertip[1])  # 좌표 정수 변환

        # 이미지에 빨간색 원(점) 그리기
        cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

# 좌표 정보 확인 (예: 첫 번째 결과)
for result in results:
    print(result.keypoints)  # keypoints: 21개 손 키포인트 좌표 출력

cv2.imshow("Fingertip Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()