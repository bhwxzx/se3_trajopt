import time
import os

import numpy as np
import pinocchio as pin

from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import *
from nltrajopt.cost_models import *
import nltrajopt.utils as reprutils


from terrain.terrain_grid import TerrainGrid

from robots.LW_Wheel.LWWheelWrapper import LW_Wheel

from visualiser.visualiser import TrajoptVisualiser

import nltrajopt.params as pars

VIS = pars.VIS
DT = 1 / 20

terrain = TerrainGrid(10, 10, 0.9, -5.0, -5.0, 5.0, 5.0)
terrain.set_zero()

robot = LW_Wheel()
q = robot.go_neutral()
q_end = robot.go_end()
q_end[0] += 0.35

contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
    "l_wheel": robot.left_wheel_frames,
    "r_wheel": robot.right_wheel_frames
}

contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)

contact_scheduler.add_phase(["l_wheel", "r_wheel"], 1.3) 
k1 = len(contact_scheduler.contact_sequence_fnames)
contact_scheduler.add_phase([], 0.35) 
k2 = len(contact_scheduler.contact_sequence_fnames)
contact_scheduler.add_phase(["l_foot", "r_foot"], 1.3) 

frame_contact_seq = contact_scheduler.contact_sequence_fnames

contact_frame_names = robot.left_foot_frames + robot.right_foot_frames + robot.left_wheel_frames + robot.right_wheel_frames

left_joints =[
    "left_hip_joint", "left_thigh_joint", "left_shank_joint", 
    "left_foot_joint", "left_wheel_joint"
]
right_joints =[
    "right_hip_joint", "right_thigh_joint", "right_shank_joint", 
    "right_foot_joint", "right_wheel_joint"
]

# 动态获取它们在优化器关节向量中的索引 (减去基座的 6 维)
l_idx = [robot.model.joints[robot.model.getJointId(name)].idx_v - 6 for name in left_joints]
r_idx =[robot.model.joints[robot.model.getJointId(name)].idx_v - 6 for name in right_joints]

# 设置对称符号。因为左右腿 Pitch 轴互为相反数，所以全是 -1
sym_signs = [-1, -1, -1, -1, -1]

# 定义 5 对关节的不对称惩罚权重
# 分别对应：Hip, Thigh, Shank, Foot, Wheel
sym_weight = np.diag([50.0, 100.0, 200.0, 1.0, 0.0])

stages = []
K = len(frame_contact_seq)
print("K = ", K)

for k, contact_phase_fnames in enumerate(frame_contact_seq):
    stage_node = Node(
        nv=robot.model.nv,
        contact_phase_fnames=contact_phase_fnames,
        contact_fnames=contact_frame_names,
    )

    dyn_const = WholeBodyDynamics() # 其中包含扭矩限制
    stage_node.dynamics_type = dyn_const.name

    stage_node.constraints_list.extend(
        [
            dyn_const,
            TimeConstraint(min_dt=DT, max_dt=DT, total_time=None),
            SemiEulerIntegration(),
            TerrainGridFrictionConstraints(terrain, max_delta_force=100.0),
            TerrainGridContactConstraints(terrain),
        ]
    )

    # ================= 分阶段设置代价函数 =================

    q_start_rep = reprutils.pin2rep(q)
    start_position = q_start_rep[0:3]
    start_rotation = q_start_rep[3:6]
    q_end_rep = reprutils.pin2rep(q_end)
    target_position = q_end_rep[0:3]
    target_rotation = q_end_rep[3:6]
    
    if k < k1:
        # 【轮式准备与起跳】
        pos_weight = np.diag([0.0, 100.0, 10.0])
        rot_weight = np.diag([200.0, 150.0, 300.0]) 
        joint_weight = np.diag([10, 0.5, 0.5, 0.1, 0.0, 10, 0.5, 0.5, 0.1, 0.0])
        force_weight = np.diag([0.0, 0.0, 2e-4])
        ref_joints = q[7:]

        stage_node.costs_list.extend([BasePositionCost(start_position, pos_weight)])
        stage_node.costs_list.extend([BaseRotVelocityCost(np.zeros(3), np.eye(3) * 1e-2)])
        stage_node.costs_list.extend([ForceCost(np.zeros(3), force_weight)])
        
    elif k1 <= k < k2:
        # 【腾空变形期】
        pos_weight = np.diag([0.0, 100.0, 0.0])
        rot_weight = np.diag([300.0, 400.0, 400.0]) 
        joint_weight = np.diag([200, 0.5, 0.5, 0.0, 0.0, 200, 0.5, 0.5, 0.0, 0.0])
        ref_joints = q_end[7:] 

        stage_node.costs_list.extend([BaseRotVelocityCost(np.zeros(3), np.eye(3) * 1e-1)])
        
    else:
        # 【足式落地与缓冲】
        pos_weight = np.diag([0.0, 100.0, 25.0])
        rot_weight = np.diag([200.0, 100.0, 0.0]) 
        joint_weight = np.diag([10, 0.01, 0.01, 0.0, 0.0, 10, 0.01, 0.01, 0.0, 0.0])
        force_weight = np.diag([0.0, 0.0, 1e-3])
        ref_joints = q_end[7:]

        # stage_node.costs_list.extend([BasePosVelocityCost(np.zeros(3), np.eye(3) * 5e-2)])
        stage_node.costs_list.extend([BasePositionCost(target_position, pos_weight)])
        stage_node.costs_list.extend([BaseRotVelocityCost(np.zeros(3), np.eye(3) * 1e-2)])
        stage_node.costs_list.extend([ForceCost(np.zeros(3), force_weight)])

    # ====================================================

    stage_node.costs_list.extend([BaseRotationCost(target_rotation, rot_weight)])

    stage_node.costs_list.extend([ConfigurationCost(ref_joints, joint_weight)])

    stage_node.costs_list.extend([JointSymmetryCost(r_idx, l_idx, sym_weight, sym_signs)])
    
    stage_node.costs_list.extend([JointVelocityCost(np.zeros((robot.model.nv - 6,)), np.eye(robot.model.nv - 6) * 1e-2)])
    stage_node.costs_list.extend([JointAccelerationCost(np.zeros(robot.model.nv - 6), np.eye(robot.model.nv - 6) * 1e-5)])

    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

opti.set_initial_pose(q)
opti.set_target_pose(q_end)

# warm start
for k, node in enumerate(opti.nodes):
    if k < k1:
        alpha = 0.0  # 起跳前，完全是初始姿态 q
    elif k <= k2:
        alpha = (k - k1) / (k2 - k1)  # 腾空期，平滑过渡
    else:
        alpha = 1.0  # 落地后，完全是目标姿态 q_end

    q_pin = pin.interpolate(robot.model, q, q_end, alpha)
    opti.x0[node.q_id] = reprutils.pin2rep(q_pin)

result = opti.solve(200, 1e-3, parallel=False, print_level=0)
opti.save_solution("wheel_to_leg")

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
vqs = [result["nodes"][k]["v"] for k in range(K)]
aqs = [result["nodes"][k]["a"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

# ======================================================================
# 轨迹后处理：生成 60Hz 的 CSV 数据
# ======================================================================
import os
import scipy.interpolate as interp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

print("\n--- 开始生成 60Hz CSV 轨迹数据 ---")

# 构建原始时间轴 t_orig 
qs_arr = np.array(qs)  # shape: (K, 17)
t_orig = np.zeros(K)
for i in range(1, K):
    t_orig[i] = t_orig[i-1] + dts[i-1]

# 构建目标时间轴 t_target (60Hz = 0.0166667s)
target_fps = 60.0
dt_target = 1.0 / target_fps
t_target = np.arange(0, t_orig[-1], dt_target)
# 确保正好包含最后一帧
if len(t_target) == 0 or t_target[-1] < t_orig[-1]:
    t_target = np.append(t_target, t_orig[-1])

# 样条插值: 基座位置 (x, y, z)
cs_pos = interp.CubicSpline(t_orig, qs_arr[:, 0:3])
pos_interp = cs_pos(t_target)

# Slerp插值: 基座四元数 (qx, qy, qz, qw)
rotations = R.from_quat(qs_arr[:, 3:7])
slerp = Slerp(t_orig, rotations)
quat_interp = slerp(t_target).as_quat()

# 动态关节映射与插值: 确保顺序严格符合 CSV 要求
target_joint_names =[
    "right_hip_joint", "left_hip_joint",
    "right_thigh_joint", "left_thigh_joint",
    "right_shank_joint", "left_shank_joint",
    "right_foot_joint", "left_foot_joint",
    "right_wheel_joint", "left_wheel_joint"
]

# 恢复使用 K，因为 qs_arr 是完整的
joint_data_orig = np.zeros((K, 10))
for i, j_name in enumerate(target_joint_names):
    if robot.model.existJointName(j_name):
        j_id = robot.model.getJointId(j_name)
        idx_q = robot.model.joints[j_id].idx_q
        joint_data_orig[:, i] = qs_arr[:, idx_q]
    else:
        print(f"[错误] URDF 中找不到关节: {j_name}，CSV数据可能出错！")

cs_joints = interp.CubicSpline(t_orig, joint_data_orig)
joints_interp = cs_joints(t_target)

# 组装最终的 60Hz 数据
csv_data = np.hstack((pos_interp, quat_interp, joints_interp))

# 在最终的 60Hz 数据中剔除头尾
if csv_data.shape[0] > 2:
    csv_data = csv_data[1:-1, :]  # 去掉第一帧(绝对起点)和最后一帧(绝对终点)

# 获取当前执行的 Python 脚本所在的绝对目录路径
script_dir = os.path.dirname(os.path.realpath(__file__))
csv_filename = os.path.join(script_dir, "wheel_to_leg_transform_60hz.csv")

# 导出为无表头的纯数据 CSV
np.savetxt(csv_filename, csv_data, delimiter=",", fmt="%.6f")
print(f"✅ 成功导出轨迹文件至: {csv_filename} (形状: {csv_data.shape})")
# ======================================================================


if VIS:
    tvis = TrajoptVisualiser(robot)

    tvis.display_robot_q(robot, qs[0])
    tvis.load_terrain(terrain)

    time.sleep(1)
    while True:
        for i in range(len(qs)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)

        tvis.update_forces(robot, {}, 0.01)



