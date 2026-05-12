# Báo cáo: Thiết kế Các Bộ Điều Khiển PID cho Quadcopter

## 1. Mục tiêu

Tài liệu này giải thích lý do tại sao các file PID được sắp xếp theo thứ tự từ 01 đến 06, cùng với công thức toán học chi tiết của từng vòng điều khiển, và phân tích những vấn đề đã được phát hiện trong quá trình xây dựng.

---

## 2. Lệnh chạy nhanh

```bash
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/01_uav_pid_altitude.py     # 1 tầng: Z → thrust

./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/02_uav_pid_rate.py

./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/03_uav_pid_attitude.py      # 1 tầng: angle → moment
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/04_uav_pid_hover.py         # 2 tầng: altitude + attitude
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/05_uav_pid_cascade.py       # 2 tầng: attitude → rate (FPV)
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/06_uav_pid_velocity.py      # 2 tầng: velocity + att (keyboard)
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/07_uav_pid_position.py      # 2 tầng: position + att
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/08_uav_pid_hierarchical.py  # 3 tầng: pos → vel → att
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/09_uav_pid_multirate.py     # 3 tầng + ZOH multi-rate
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/10_uav_pid_navigation.py    # 3 tầng + waypoints
```

---

## 3. Tổng quan thứ tự — Độ khó tăng dần

| File | Tên             | Tầng | Mô tả                           |
| ---- | ---------------- | ----- | --------------------------------- |
| 01   | `altitude`     | 1     | Z → thrust                       |
| 02   | `rate`         | 1     | p/q/r → moment (innermost)       |
| 03   | `attitude`     | 1     | angle → moment                   |
| 04   | `hover`        | 2     | altitude + attitude song song     |
| 05   | `cascade`      | 2     | attitude → rate (FPV/Betaflight) |
| 06   | `velocity`     | 2     | velocity + attitude (keyboard)    |
| 07   | `position`     | 2     | position + attitude               |
| 08   | `hierarchical` | 3     | pos → vel → att                 |
| 09   | `multirate`    | 3+ZOH | multi-rate, PID_freq ≠ sim_freq  |
| 10   | `navigation`   | 3+WP  | hierarchical + waypoints          |

---

## 3. Phân tích chi tiết từng file

### 3.1 File 01 — Altitude Only (`01_uav_pid_altitude.py`)

**Mục tiêu:** Chỉ điều khiển độ cao Z, giữ X/Y cố định.

**Điểm mới học được:** Cấu trúc cơ bản nhất của PID drone: 1 vòng ngoài + 1 vòng trong.

**Sơ đồ:**

```
z_target ─► [PID_z] ─► thrust (N)
                              │
                              ▼
              [Quadcopter Physics]
                              ▲
roll=0   ─► [PID_roll ] ─► m_roll  (Nm)
pitch=0  ─► [PID_pitch] ─► m_pitch (Nm)
yaw=0    ─► [PID_yaw  ] ─► m_yaw   (Nm)
```

**Công thức:**

Vòng ngoài (altitude):

```
e_z = z_target - z
thrust = kp_z · e_z + ki_z · ∫e_z dt + kd_z · (de_z/dt)
thrust = max(0, thrust)   ← không âm
```

Vòng trong (attitude — giữ drone level):

```
e_roll  = 0 - roll
e_pitch = 0 - pitch
e_yaw   = normalize(0 - yaw)   ← normalize về [-π, π]

m_roll  = kp_r · e_roll  + ki_r · ∫e_roll  dt + kd_r · (de_roll/dt)
m_pitch = kp_p · e_pitch + ki_p · ∫e_pitch dt + kd_p · (de_pitch/dt)
m_yaw   = kp_y · e_yaw   + kd_y · (de_yaw/dt)
```

**Lưu ý:** Không dùng feedforward trọng lực → integral ki_z tích lũy để bù mg theo thời gian (chậm hơn nhưng không cần biết khối lượng).

---

### 3.2 File 02 — Position Control 3D (`02_uav_pid_position.py`)

**Mục tiêu:** Bay đến vị trí (x, y, z) mục tiêu.

**Điểm mới học được:** Mở rộng từ 1D lên 3D. Coupling giữa position (x/y) và attitude (roll/pitch) — muốn bay theo X thì phải nghiêng (pitch), muốn bay theo Y thì phải lăn (roll).

**Sơ đồ:**

```
                    ┌─ [PID_z] ─► thrust (N)
                    │
target_pos ─► pos_error ─┤
                    │
                    └─ [PID_x] ─► desired_pitch (rad) ──┐
                      [PID_y] ─► desired_roll  (rad) ──┤
                                                         │
                    ┌────────────────────────────────────┘
                    │
                    ▼
         [PID_roll ](des_roll  - roll ) ─► m_roll  (Nm)
         [PID_pitch](des_pitch - pitch) ─► m_pitch (Nm)
         [PID_yaw  ](0         - yaw  ) ─► m_yaw   (Nm)
```

**Công thức:**

Vòng ngoài (position → thrust + desired attitude):

```
e_z = z_t - z;  thrust     = max(0, PID_z(e_z))
e_x = x_t - x;  des_pitch  = clamp(-PID_x(e_x), ±20°)
e_y = y_t - y;  des_roll   = clamp( PID_y(e_y), ±20°)
```

> **Tại sao des_pitch = -PID_x?**
> Bay về phía +X cần drone nghiêng về phía trước → pitch âm (theo convention).
> Bay về phía +Y cần drone nghiêng sang phải → roll dương.

Vòng trong (attitude tracking):

```
m_roll  = PID_roll (des_roll  - roll )
m_pitch = PID_pitch(des_pitch - pitch)
m_yaw   = PID_yaw  (0 - yaw)
```

---

### 3.3 File 03 — Velocity Control (`03_uav_pid_velocity.py`)

**Mục tiêu:** Điều khiển **vận tốc** (vx, vy, vz) thay vì vị trí. Keyboard interactive.

**Điểm mới học được:** Paradigm điều khiển khác hoàn toàn. Không có tham chiếu vị trí tuyệt đối — drone giữ vận tốc mục tiêu nhưng có thể drift. Hữu ích cho joystick/RC control.

**Sơ đồ:**

```
TARGET_VEL [vx,vy,vz] (từ bàn phím)
         │
         ├─ [PID_vz](TARGET_vz - vz) ─► thrust (N)
         │
         ├─ [PID_vx](TARGET_vx - vx) ─► desired_pitch (rad) ──┐
         └─ [PID_vy](TARGET_vy - vy) ─► desired_roll  (rad) ──┤
                                                                │
                                          [PID_roll/pitch/yaw] ┘
                                               ─► moments (Nm)
```

**Công thức:**

Vòng ngoài (velocity → thrust + desired attitude):

```
e_vz = vz_target - vz;  thrust    = max(0, PID_vz(e_vz))
e_vx = vx_target - vx;  des_pitch = clamp(-PID_vx(e_vx), ±20°)
e_vy = vy_target - vy;  des_roll  = clamp( PID_vy(e_vy), ±20°)
```

**So sánh với file 02:**

- File 02: `e_z = z_target - z` (error trên vị trí)
- File 03: `e_vz = vz_target - vz` (error trên vận tốc)
- Integral của PID_vz tích lũy → tự bù trọng lực để hover khi vz_target = 0

---

### 3.4 File 04 — Multi-rate Control (`04_uav_pid_hover.py`)

**Mục tiêu:** Position control giống file 02, nhưng PID chạy ở **tần số thấp hơn** tần số sim.

**Điểm mới học được:** Trong thực tế, vi xử lý điều khiển (50-200 Hz) chạy chậm hơn physics engine (200-1000 Hz). File này mô phỏng đúng ràng buộc đó.

**Sơ đồ thời gian:**

```
sim tick: │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
          │──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──│
           200 Hz                                    200 Hz

PID tick: │           │           │           │           │
          │───────────┴───────────┴───────────┴───────────│
           50 Hz                                      50 Hz
```

**Công thức:**

```python
CONTROL_HZ = 50           # PID chạy 50 lần/giây
SIM_HZ     = 200          # Physics 200 lần/giây
DECIMATION = SIM_HZ // CONTROL_HZ  # = 4

if step % DECIMATION == 0:
    dt = SIM_DT * DECIMATION    # dt đúng cho PID
    thrust, des_roll, des_pitch = outer_PID(pos, target, dt)
    m_roll, m_pitch, m_yaw      = inner_PID(att, des_att, dt)

# Apply lực MỖI sim step (200 Hz) với giá trị PID cũ (ZOH)
apply_wrench(thrust, moments)
```

**ZOH (Zero-Order Hold):** Giữ nguyên lệnh điều khiển cho đến khi PID chạy lần tiếp theo.

**Tại sao đặt sau file 03?**
File 02 và 03 giả định PID chạy mỗi sim step. File 04 mới đưa ra câu hỏi: nếu phần cứng chỉ chạy được 50 Hz, ta làm thế nào? → thêm decimation logic.

---

### 3.5 File 05 — Hierarchical 3-tier Cascade (`05_uav_pid_hierarchical.py`)

**Mục tiêu:** Kiến trúc 3 tầng đúng như firmware quadcopter thực tế.

**Điểm mới học được:** Cascade nhiều tầng với tần số khác nhau. Mỗi tầng có dải thông (bandwidth) riêng.

**Sơ đồ:**

```
┌──────────────────────────────────────────────┐
│  Tầng 1 — Position PID (10 Hz)               │
│  e_p = target - pos                          │
│  des_vel = K_p · e_p + K_i · ∫e_p + K_d · ė_p│
│  clamp(des_vel, ±MAX_VEL)                    │
└─────────────────────┬────────────────────────┘
                      │ desired_velocity [vx,vy,vz]
                      ▼
┌──────────────────────────────────────────────┐
│  Tầng 2 — Velocity PID (25 Hz)               │
│  e_v = des_vel - vel                         │
│  thrust   = max(0, PID_vz(e_vz))            │
│  des_pitch = -clamp(PID_vx(e_vx), ±20°)    │
│  des_roll  =  clamp(PID_vy(e_vy), ±20°)    │
└─────────────────────┬────────────────────────┘
                      │ thrust, des_roll, des_pitch
                      ▼
┌──────────────────────────────────────────────┐
│  Tầng 3 — Attitude PID (200 Hz = mỗi step)  │
│  m_roll  = PID_r(des_roll  - roll )          │
│  m_pitch = PID_p(des_pitch - pitch)          │
│  m_yaw   = PID_y(0 - yaw)                   │
└──────────────────────────────────────────────┘
```

**Công thức tầng 1 (Position → Velocity):**

```
Δt₁ = 1/10 s = 0.1 s

des_vx = kp_px·(xt-x) + ki_px·∫(xt-x)dt + kd_px·d(xt-x)/dt
des_vy = kp_py·(yt-y) + ki_py·∫(yt-y)dt + kd_py·d(yt-y)/dt
des_vz = kp_pz·(zt-z) + ki_pz·∫(zt-z)dt + kd_pz·d(zt-z)/dt
```

**Công thức tầng 2 (Velocity → Thrust + Angles):**

```
Δt₂ = 1/25 s = 0.04 s

thrust    = max(0, kp_vz·(des_vz-vz) + ki_vz·∫... + kd_vz·...)
des_pitch = clamp(-PID_vx(des_vx-vx), ±20°)
des_roll  = clamp( PID_vy(des_vy-vy), ±20°)
```

**Công thức tầng 3 (Attitude → Moments):**

```
Δt₃ = 0.005 s = 200 Hz

m_roll  = kp_r·(des_roll -roll ) + ki_r·∫... + kd_r·...
m_pitch = kp_p·(des_pitch-pitch) + ki_p·∫... + kd_p·...
m_yaw   = kp_y·(0-yaw) + kd_y·...
```

**Tại sao 3 tần số khác nhau?**

| Tầng    | Tần số | Lý do                                                   |
| -------- | -------- | -------------------------------------------------------- |
| Position | 10 Hz    | Động học vị trí chậm, không cần tính liên tục |
| Velocity | 25 Hz    | Cần phản hồi nhanh hơn position                      |
| Attitude | 200 Hz   | Động học góc nhanh nhất, phải chạy mỗi step      |

---

### 3.6 File 06 — Navigation with Hierarchical Control (`06_uav_pid_navigation.py`)

**Mục tiêu:** Ứng dụng kiến trúc 3-tầng (file 05) vào bài toán điều hướng waypoint ngẫu nhiên.

**Điểm mới học được:** Kết hợp bộ điều khiển phức tạp nhất (hierarchical) với logic cấp cao hơn (waypoint switching). Đây là dạng gần với hệ thống thực nhất.

**Sơ đồ tổng thể:**

```
[Mission Planner]
   - Sample waypoint ngẫu nhiên
   - Switch khi err < threshold hoặc timeout

        │ target_pos
        ▼
[Hierarchical 3-tier PID]  (giống file 05)
        │ thrust, moments
        ▼
[UAV Physics]
        │ pos, vel, att
        └──────────────────► feedback về 2 tầng trên
```

---

## 4. Vấn đề phát hiện và sửa đổi

### Vấn đề 1: File 04 là bản sao của file 02

**Phát hiện:** File 04 (hover) dùng cùng thuật toán với file 02 (position) — cùng `QuadcopterPID` outer + attitude inner. Không dạy thêm khái niệm mới.

**Sửa:** Rewrite file 04 để nhấn mạnh rõ ràng **multi-rate** là điểm khác biệt — PID chạy ở CONTROL_HZ (50 Hz), sim chạy ở SIM_HZ (200 Hz). Thêm comment giải thích `DECIMATION` và ZOH.

### Vấn đề 2: File 06 bước lùi về độ phức tạp

**Phát hiện:** File 06 (sau khi cập nhật ban đầu) dùng `QuadcopterPID` (2-loop, đơn giản hơn) thay vì hierarchical 3-loop của file 05. Như vậy file 06 **đơn giản hơn** file 05 — vi phạm nguyên tắc tăng độ khó.

**Sửa:** Rewrite file 06 dùng **hierarchical 3-tier control** (giống file 05) cho bài toán navigation, làm cho nó thực sự là bước tiếp theo của file 05.

### Vấn đề 3: Không dùng feedforward trọng lực

**Quyết định có chủ ý:** Tất cả file đều bỏ `hover_thrust` feedforward để PID hoạt động mà không cần biết khối lượng thực tế.

**Hệ quả:** Drone cần vài giây đầu để integral tích lũy bù trọng lực. Rise time chậm hơn so với có feedforward, nhưng hoàn toàn không phụ thuộc thông số vật lý đo được.

---

## 5. Bảng thống kê PID Controllers trong từng file

| File | PID instances | Outer loop                        | Inner loop                   |
| ---- | ------------- | --------------------------------- | ---------------------------- |
| 01   | 4             | PID_z (1D)                        | PID_roll, PID_pitch, PID_yaw |
| 02   | 6             | PID_z, PID_x, PID_y (3D)          | PID_roll, PID_pitch, PID_yaw |
| 03   | 6             | PID_vz, PID_vx, PID_vy (velocity) | PID_roll, PID_pitch, PID_yaw |
| 04   | 6             | PID_z, PID_x, PID_y (decimated)   | PID_roll, PID_pitch, PID_yaw |
| 05   | 9             | PID_px/py/pz + PID_vx/vy/vz       | PID_roll, PID_pitch, PID_yaw |
| 06   | 9             | PID_px/py/pz + PID_vx/vy/vz       | PID_roll, PID_pitch, PID_yaw |

---

## 6. Thứ tự học đề xuất

```
01 → 02 → 03 → 04 → 05 → 06
 │    │    │    │    │    │
 │    │    │    │    │    └─ Navigation: hierarchical + waypoints
 │    │    │    │    └────── Hierarchical: 3-tier cascade
 │    │    │    └─────────── Multi-rate: PID_freq ≠ sim_freq
 │    │    └──────────────── Velocity: vel-based paradigm
 │    └───────────────────── Position 3D: 2-loop cascade
 └────────────────────────── Altitude: 1D, đơn giản nhất
```
