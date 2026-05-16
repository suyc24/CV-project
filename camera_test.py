import cv2
import os

# 创建保存目录
save_dir = "camera_captures"
os.makedirs(save_dir, exist_ok=True)

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    
    if ret:
        print(f"Camera {i} works: {frame.shape}")
        # 保存图片
        filename = os.path.join(save_dir, f"camera_{i}.jpg")
        cv2.imwrite(filename, frame)
        print(f"  -> Saved to {filename}")
    else:
        print(f"Camera {i} failed")
    
    cap.release()

print(f"\nDone! Check the '{save_dir}' folder for captured images.")


for idx in range(6):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        continue
        
    print(f"\n===== CAMERA INDEX {idx} =====")
    # 遍历常用的FourCC编码格式
    formats = [('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
               ('YUYV', cv2.VideoWriter_fourcc(*'YUYV')),
               ('H264', cv2.VideoWriter_fourcc(*'H264')),
               ('NV12', cv2.VideoWriter_fourcc(*'NV12'))]
    
    for name, fourcc in formats:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        ret, frame = cap.read()
        if ret and frame is not None and frame.mean() > 1:  # 非全黑
            print(f"  ✅ {name} WORKS! Shape: {frame.shape}")
            # 保存一张作为凭证
            cv2.imwrite(f"{save_dir}/test_camera_{idx}_{name}.jpg", frame)
        else:
            print(f"  ❌ {name} fails or returns black frame")
    cap.release()