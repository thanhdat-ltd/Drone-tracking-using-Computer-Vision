### Kiến trúc wrench trong Isaac Lab

`permanent_wrench_composer.set_forces_and_torques()` áp lực/momen theo **LOCAL frame** của từng body.

```
Crazyflie articulation:
  root: "body"          ← thân drone chính
    ├── m1_prop (CCW)   ← revolute joint, trục Z tự do
    ├── m2_prop (CW)
    ├── m3_prop (CCW)
    └── m4_prop (CW)
```

---

### Quy tắc truyền lực qua revolute joint

| Wrench áp lên prop body                    | Truyền lên root? | Lý do                                                   |
| -------------------------------------------- | ------------------ | -------------------------------------------------------- |
| `forces[..., 2]` — thrust dọc trục spin | **Có**      | Trục Z bị ràng buộc về vị trí → lực truyền qua |
| `torques[..., 2]` — spin torque           | **Không**   | Trục Z là DOF tự do của joint → không ràng buộc  |
| `torques[..., 0/1]` — lật ngang          | **Có**      | Trục X/Y bị ràng buộc → truyền qua                 |

---

### 3 kênh điều khiển — hoàn toàn độc lập

#### Kênh 1: Thrust (truyền lên root)

```python
forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
forces[..., 2] = F_per_motor           # lực đẩy lên (N)
```

#### Kênh 2: Spin props (KHÔNG truyền lên root)

```python
# m1 CCW(+), m2 CW(-), m3 CCW(+), m4 CW(-)
prop_spin_dirs = torch.tensor([1.0, -1.0, 1.0, -1.0], device=sim.device)
torques = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
torques[..., 2] = prop_spin_dirs * SPIN_TORQUE    # Nm, tăng để props quay nhanh hơn
```

Gọi chung một lần:

```python
robot.permanent_wrench_composer.set_forces_and_torques(
    forces=forces, torques=torques, body_ids=prop_body_ids,
)
```

#### Kênh 3: Yaw torque (áp thẳng lên root, bypass joint)

```python
# net yaw = 4 × CT_RATIO × yaw_frac × F_base
CT_RATIO = 0.005971   # Crazyflie torque-to-thrust ratio (m)
net_yaw = torch.tensor(
    [[0.0, 0.0, 4.0 * CT_RATIO * yaw_frac * F_base]],
    device=sim.device,
).expand(robot.num_instances, -1).unsqueeze(1)

robot.permanent_wrench_composer.add_forces_and_torques(   # add, không set
    forces=torch.zeros_like(net_yaw),
    torques=net_yaw,
    body_ids=root_body_ids,
)
```

> **`set` vs `add`** : dùng `set` cho props (ghi đè mỗi step), dùng `add` cho root (cộng dồn vào wrench đã có từ props).

---

### Tạo yaw bằng differential thrust (giống firmware thực)

```
F_ccw (m1, m3) = F_base × (1 − yaw_frac)
F_cw  (m2, m4) = F_base × (1 + yaw_frac)

Tổng thrust = 4 × F_base = m × g  ✓
Net yaw     = 4 × CT_RATIO × yaw_frac × F_base  ✓
```

`yaw_frac ∈ [0, 1]` — giá trị 0.3 là hợp lý cho Crazyflie.

---

### Setup đúng trước vòng lặp

```python
prop_body_ids = robot.find_bodies("m.*_prop")[0]
root_body_ids = robot.find_bodies("body")[0]      # in body_names nếu không chắc tên
robot_mass    = robot.root_physx_view.get_masses().sum()
gravity       = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

CT_RATIO       = 0.005971
prop_spin_dirs = torch.tensor([1.0, -1.0, 1.0, -1.0], device=sim.device)
SPIN_TORQUE    = 0.01   # Nm
```

---

### Thứ tự gọi mỗi step

```python
# 1. Thrust + spin torque → prop bodies
robot.permanent_wrench_composer.set_forces_and_torques(...)

# 2. Yaw torque → root body
robot.permanent_wrench_composer.add_forces_and_torques(...)

# 3. Commit xuống PhysX
robot.write_data_to_sim()

# 4. Step physics
sim.step()

# 5. Cập nhật buffer
robot.update(sim_dt)
```
