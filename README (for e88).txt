Giai đoạn 1 — Test hoàn toàn không cần drone (--no-fly)
Mục tiêu: xác nhận YOLO detect đúng, dashboard hiển thị đúng, PID tính đúng chiều.

Bước 1: Kiểm tra dependencies
pip install ultralytics opencv-python

Bước 2: Chạy với webcam
cd "<đường dẫn>\src"
python main_tracking.py --no-fly --source 0 --flip
python main_tracking.py --no-fly --source 0 --flip --model <model>.pt

Quan sát trên dashboard:
Bộ xương (skeleton) vẽ đúng lên người không?
err_x và err_y gần 0 khi bạn đứng giữa frame?
Khi bước sang phải: err_x tăng dương → cột YAW thay đổi?
Khi lùi ra xa: shoulder_span giảm → cột PITCH thay đổi?
Nhấn T → State chuyển FLYING, đợi 2 giây, sau đó chuyển TRACKING?
Nhấn E ngay sau T → State về IDLE ngay không?


Giai đoạn 2 — Kết nối drone, test stream và lệnh (chưa cất cánh)
Mục tiêu: xác nhận WiFi, RTSP, và lệnh UDP gửi được — không takeoff.

Bước 1: Kết nối WiFi drone
Bật drone, kết nối máy tính vào WiFi FLOW-UFO-XXXXXX

Bước 2: Test stream video
python main_tracking.py --no-fly
python main_tracking.py --no-fly --model yolov8l-pose.pt	
Nếu thấy video từ camera drone → stream OK.
Nếu không thấy: kiểm tra ping 192.168.1.1

Bước 3: Test gửi lệnh UDP (drone vẫn đặt trên mặt đất)


python main_tracking.py
python main_tracking.py --model <model>.pt
# KHÔNG nhấn T — chỉ quan sát kết nối
Log phải hiện: Connected to FLOW-UFO at 192.168.1.1:7099

Giai đoạn 3 — Bay thật lần đầu
Chuẩn bị không gian:

Phòng rộng rãi, dọn chướng ngại vật vì drone không né được vật cản
Bạn đứng cách drone ~1.5–2m để YOLO nhận diện được người

Trình tự bay:
1. Chạy: python main_tracking.py
2. Chờ log "Drone connected" và thấy video stream
3. Đứng vào khung hình (không nhất thiết phải thấy người)
4. Nhấn T → drone cất cánh
5. Quan sát 2 giây: drone phải HOVER yên (State = FLYING)
6. Sau 2s: TRACKING → drone bắt đầu điều chỉnh (phải thấy người)
7. Thử bước chậm sang trái/phải → drone xoay theo
8. Nhấn H bất cứ lúc nào để drone hover lại
9. Nhấn T lần nữa để hạ cánh
Phím khẩn cấp luôn sẵn sàng:

Phím	Tác dụng
H	Hover ngay, PID reset
T	Hạ cánh an toàn
E	Tắt motor ngay lập tức
Q	Thoát + hạ cánh
Nếu drone phản ứng quá mạnh/yếu sau khi bay thật
Chỉnh trong pid_controller.py


# Drone xoay quá nhanh → giảm kp yaw
# Drone tiến/lùi quá mạnh → giảm kp pitch
# Drone lên/xuống giật → giảm kp throttle
Mỗi lần chỉnh chỉ thay đổi 1 trục, bay thử lại để đánh giá trước khi chỉnh tiếp.