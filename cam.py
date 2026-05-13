import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    if ret:
        print(f"Camera {i} works: {frame.shape}")
    else:
        print(f"Camera {i} failed")
    cap.release()