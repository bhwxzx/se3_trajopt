"""60 Hz LW leg-to-wheel trajectory optimization with direct solver-node CSV export."""

import time
import os

import numpy as np
import pinocchio as pin

from nltrajopt.trajectory_optimization import NLTrajOpt  # export PYTHONPATH=$(pwd)/src
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import (
    SemiEulerIntegration,
    TerrainGridContactConstraints,
    TerrainGridFrictionConstraints,
    TimeConstraint,
    WholeBodyDynamics,
)
from nltrajopt.cost_models import (
    BasePositionCost,
    BasePosVelocityCost,
    BaseRotationCost,
    BaseRotVelocityCost,
    ConfigurationCost,
    ForceCost,
    JointAccelerationCost,
    JointSymmetryCost,
    JointVelocityCost,
)
import nltrajopt.utils as reprutils

from terrain.terrain_grid import TerrainGrid

from robots.LW_Leg.LWLegWrapper import LW_Leg

from visualiser.visualiser import TrajoptVisualiser

import nltrajopt.params as pars

VIS = pars.VIS
DT = 1 / 60

def build_problem(q_end_override=None, q_start_override=None, wheel_contact=False):
    """Build the leg-to-wheel problem, optionally replacing its boundary poses."""
    terrain = TerrainGrid(10, 10, 0.9, -5.0, -5.0, 5.0, 5.0)
    terrain.set_zero()

    robot = LW_Leg()
    wheels = None
    if wheel_contact:
        from nltrajopt.wheel_contact import WheelContactModel
        from nltrajopt.constraint_models.wheel_contact import RollingWheelContactConstraints, RollingWheelDynamics
        wheels = WheelContactModel(robot.model)
    q = robot.go_neutral()
    # 初态：调整高度和足角，使八个足底接触点位于 z=0。
    q[2] = 0.6948750854291303
    initial_foot_angles = {
        "left_foot_joint": 0.4712926535897933,
        "right_foot_joint": -0.4712926535897933,
    }
    for joint_name, angle in initial_foot_angles.items():
        joint_id = robot.model.getJointId(joint_name)
        q[robot.model.joints[joint_id].idx_q] = angle
    if q_start_override is not None:
        q = np.asarray(q_start_override, dtype=float).copy()
        if q.shape != (robot.model.nq,):
            raise ValueError("Initial pose must use the model Pinocchio q ordering")
    q_end = robot.go_end()
    q_end[0] += -0.45   # 0.0 -0.15 -0.3 -0.45 -0.6
    # 滚动接触模型终态（暂不启用；恢复模型时切换这两组终态参数）。
    # q_end[2] = 0.6718648897273728
    # end_pitch_rad = np.deg2rad(-0.21511134770760826)
    # q_end[3:7] = [0.0, np.sin(end_pitch_rad / 2), 0.0, np.cos(end_pitch_rad / 2)]

    # 旧固定接触点终态：两轮虚拟接触点位于 z=0，COM 投影位于支撑线上。
    q_end[2] = 0.6716482428273907
    end_pitch_rad = np.deg2rad(-0.1666406670479646)
    q_end[3:7] = [0.0, np.sin(end_pitch_rad / 2), 0.0, np.cos(end_pitch_rad / 2)]
    if q_end_override is not None:
        q_end = np.asarray(q_end_override, dtype=float).copy()
        if q_end.shape != (robot.model.nq,):
            raise ValueError("Terminal pose must use the model Pinocchio q ordering")

    contacts_dict = {
        "l_foot": robot.left_foot_frames,
        "r_foot": robot.right_foot_frames,
        "l_wheel": robot.left_wheel_frames,
        "r_wheel": robot.right_wheel_frames
    }

    contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)

    contact_scheduler.add_phase(["l_foot", "r_foot"], 1.3)
    k1 = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase([], 0.25)
    k2 = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase(["l_wheel", "r_wheel"], 1.8)

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
    sym_weight = np.diag([50.0, 100.0, 100.0, 1.0, 0.0])

    stages = []

    for k, contact_phase_fnames in enumerate(frame_contact_seq):
        stage_node = Node(
            nv=robot.model.nv,
            contact_phase_fnames=contact_phase_fnames,
            contact_fnames=contact_frame_names,
        )

        dyn_const = RollingWheelDynamics(wheels) if wheels else WholeBodyDynamics()
        stage_node.dynamics_type = dyn_const.name

        stage_node.constraints_list.extend(
            [
                dyn_const,
                TimeConstraint(min_dt=DT, max_dt=DT, total_time=None),
                SemiEulerIntegration(),
                TerrainGridFrictionConstraints(terrain, max_delta_force=100.0),
                RollingWheelContactConstraints(terrain, wheels) if wheels else TerrainGridContactConstraints(terrain),
            ]
        )

        # ================= 分阶段设置代价函数 =================

        q_start_rep = reprutils.pin2rep(q)
        start_position = q_start_rep[0:3]
        q_end_rep = reprutils.pin2rep(q_end)
        target_position = q_end_rep[0:3]
        target_rotation = q_end_rep[3:6]

        if k < k1:
            # 【足式准备与起跳】
            pos_weight = np.diag([100.0, 100.0, 0.0])
            rot_weight = np.diag([100.0, 100.0, 10.0])
            joint_weight = np.diag([10, 0.01, 0.01, 0.0, 0.0, 10, 0.01, 0.01, 0.0, 0.0])
            force_weight = np.diag([0.0, 0.0, 1e-3])
            ref_joints = q[7:]

            stage_node.costs_list.extend([BasePositionCost(start_position, pos_weight)])
            stage_node.costs_list.extend([BaseRotVelocityCost(np.zeros(3), np.eye(3) * 2e-1)])
            stage_node.costs_list.extend([ForceCost(np.zeros(3), force_weight)])

        elif k1 <= k < k2:
            # 【腾空变形期】
            pos_weight = np.diag([0.0, 100.0, 0.0])
            rot_weight = np.diag([10.0, 100.0, 10.0])
            joint_weight = np.diag([10, 0.01, 0.01, 0.0, 0.0, 10, 0.01, 0.01, 0.0, 0.0])
            ref_joints = q_end[7:]

            stage_node.costs_list.extend([BaseRotVelocityCost(np.zeros(3), np.eye(3) * 1e-1)])

        else:
            # 【轮式落地与缓冲】
            pos_weight = np.diag([0.0, 100.0, 10.0])
            rot_weight = np.diag([100.0, 100.0, 50.0])
            joint_weight = np.diag([10, 0.5, 0.5, 0.0, 0.0, 10, 0.5, 0.5, 0.0, 0.0])
            # 4-19 历史版本未启用落地接触力代价，保留设置供后续恢复。
            # 接触力为局部坐标；等向权重惩罚合力大小，避免混淆局部 z 与世界竖直方向。
            # landing_force_weight = np.eye(3) * 5e-4
            ref_joints = q_end[7:]

            stage_node.costs_list.extend([BasePosVelocityCost(np.zeros(3), np.eye(3) * 5e-2)])
            stage_node.costs_list.extend([BasePositionCost(target_position, pos_weight)])
            stage_node.costs_list.extend([BaseRotVelocityCost(np.zeros(3), np.eye(3) * 1e-1)])

            # stage_node.costs_list.extend([ForceCost(np.zeros(3), landing_force_weight)])

        # ====================================================

        stage_node.costs_list.extend([BaseRotationCost(target_rotation, rot_weight)])

        stage_node.costs_list.extend([ConfigurationCost(ref_joints, joint_weight)])

        stage_node.costs_list.extend([JointSymmetryCost(r_idx, l_idx, sym_weight, sym_signs)])

        stage_node.costs_list.extend([JointVelocityCost(np.zeros((robot.model.nv - 6,)), np.eye(robot.model.nv - 6) * 5e-2)])
        stage_node.costs_list.extend([JointAccelerationCost(np.zeros(robot.model.nv - 6), np.eye(robot.model.nv - 6) * 1e-4)])

        stages.append(stage_node)

    opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)
    if wheels:
        opti.wheel_contact_model = wheels

    opti.set_initial_pose(q)
    opti.set_target_pose(q_end)

    # warm start：腾空段插值，落地后的轮支撑段统一使用终态轮式姿态。
    q_end_guess_opt = reprutils.pin2rep(q_end)
    for k, node in enumerate(opti.nodes):
        if k1 <= k < k2:
            alpha = (k - k1) / (k2 - k1)

            q_pin = pin.interpolate(robot.model, q, q_end, alpha)

            q_guess_opt = reprutils.pin2rep(q_pin)

            opti.x0[node.q_id] = q_guess_opt
        elif k >= k2:
            opti.x0[node.q_id] = q_end_guess_opt

    # 最后 0.2 秒：12 个时间间隔、13 个节点，保持零速度。
    hold_nodes = int(round(0.2 / DT)) + 1
    zero_velocity = [0.0] * robot.model.nv
    for node in opti.nodes[-hold_nodes:]:
        opti.lb[node.vq_id] = zero_velocity.copy()
        opti.ub[node.vq_id] = zero_velocity.copy()
        opti.x0[node.vq_id] = 0.0

    # 两足关节在最后 0.5 秒静止；结合零角度终态和积分约束，保持足角归零。
    foot_zero_nodes = int(round(0.5 / DT)) + 1
    foot_velocity_indices = [
        robot.model.joints[robot.model.getJointId(name)].idx_v
        for name in ("right_foot_joint", "left_foot_joint")
    ]
    for node in opti.nodes[-foot_zero_nodes:]:
        for index in foot_velocity_indices:
            velocity_id = node.vq_id.start + index
            opti.lb[velocity_id] = opti.ub[velocity_id] = 0.0
            opti.x0[velocity_id] = 0.0

    return opti, robot, terrain

def main():
    opti, robot, terrain = build_problem()
    result = opti.solve(800, 1e-3, parallel=False, print_level=0)
    opti.save_solution("leg_to_wheel")

    K = len(result["nodes"])
    dts = [result["nodes"][k]["dt"] for k in range(K)]
    qs = [result["nodes"][k]["q"] for k in range(K)]
    forces = [result["nodes"][k]["forces"] for k in range(K)]

    # ======================================================================
    # 直接导出 60Hz 求解节点，不对平移、四元数或关节进行插值。
    # ======================================================================
    print("\n--- 开始直接导出 60Hz 求解节点 CSV（不插值）---")

    qs_arr = np.array(qs)  # shape: (K, 17)

    # 按模型 idx_q 提取节点关节角，保持现有 CSV 的关节顺序。
    target_joint_names =[
        "right_hip_joint", "left_hip_joint",
        "right_thigh_joint", "left_thigh_joint",
        "right_shank_joint", "left_shank_joint",
        "right_foot_joint", "left_foot_joint",
        "right_wheel_joint", "left_wheel_joint"
    ]

    joint_data = np.zeros((K, 10))
    for i, j_name in enumerate(target_joint_names):
        if robot.model.existJointName(j_name):
            j_id = robot.model.getJointId(j_name)
            idx_q = robot.model.joints[j_id].idx_q
            joint_data[:, i] = qs_arr[:, idx_q]
        else:
            raise ValueError(f"URDF 中找不到关节: {j_name}，无法直接导出 CSV")

    # 平移与四元数直接取自节点 q，保留全部节点及真实首末帧。
    csv_data = np.hstack((qs_arr[:, 0:7], joint_data))

    # 获取当前执行的 Python 脚本所在的绝对目录路径
    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_filename = os.path.join(script_dir, "leg_to_wheel_transform_60hz.csv")

    # 导出为无表头的纯数据 CSV
    np.savetxt(csv_filename, csv_data, delimiter=",", fmt="%.6f")
    print(f"✅ 成功导出轨迹文件至: {csv_filename} (形状: {csv_data.shape})")
    print(f"节点间隔: {DT:.9f}s，末节点时间: {sum(dts[:-1]):.6f}s")
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

if __name__ == "__main__":
    main()
