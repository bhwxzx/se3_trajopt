import os
import numpy as np  # Linear Algebra

import pinocchio as pin  # Pinocchio library

# import example_robot_data

from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.shortcuts import buildModelsFromUrdf
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g


class LW_Leg:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(dir_path, "LW_Leg/urdf/LW.urdf")
        model_path = os.path.join(dir_path, "LW_Leg")

        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, package_dirs=[model_path],root_joint=pin.JointModelFreeFlyer()
        )

        self.data = self.model.createData()

        self.left_foot_frames = [
            "left_foot_upper_left",
            "left_foot_upper_right",
            "left_foot_lower_left",
            "left_foot_lower_right",
        ]
        self.right_foot_frames = [
            "right_foot_upper_left",
            "right_foot_upper_right",
            "right_foot_lower_left",
            "right_foot_lower_right",
        ]
        self.right_wheel_frames = ["right_wheel_contact_link"]
        self.left_wheel_frames = ["left_wheel_contact_link"]

    def fk_all(self, q, v=None):
        if v is not None:
            pin.forwardKinematics(
                self.model, self.data, q, v
            )  # FK and Forward Velocities
        else:
            pin.forwardKinematics(self.model, self.data, q)  # FK
        pin.updateFramePlacements(self.model, self.data)  # Update frames

    def go_neutral(self):
        q = pin.neutral(self.model)
        
        # 设置基座高度
        q[2] = 0.6960
        
        target_joint_angles = {
            "right_hip_joint": 0.0,       "left_hip_joint": 0.0,
            "right_thigh_joint": -0.4363, "left_thigh_joint": 0.4363,
            "right_shank_joint": -2.234,  "left_shank_joint": 2.234,
            "right_wheel_joint": 0.0,     "left_wheel_joint": 0.0,
            "right_foot_joint": -0.4712,  "left_foot_joint": 0.4712 
        }

        # 遍历并赋值
        for joint_name, angle in target_joint_angles.items():
            if self.model.existJointName(joint_name):
                joint_id = self.model.getJointId(joint_name)
                # 直接赋值，因为现在所有关节都是 1 维的 (nq=1)
                idx_q = self.model.joints[joint_id].idx_q
                q[idx_q] = angle
            else:
                print(f"[警告] URDF 中找不到关节: '{joint_name}'")

        self.fk_all(q)
        return q
    
    def go_end(self):
        q = pin.neutral(self.model)
        
        # 设置基座高度
        q[2] = 0.6720
        
        target_joint_angles = {
            "right_hip_joint": 0.0,       "left_hip_joint": 0.0,
            "right_thigh_joint": 0.7330,  "left_thigh_joint": -0.7330,
            "right_shank_joint": -0.1745, "left_shank_joint": 0.1745,
            "right_wheel_joint": 0.0,     "left_wheel_joint": 0.0,
            "right_foot_joint": 0.0,      "left_foot_joint": 0.0
        }

        # 遍历并赋值
        for joint_name, angle in target_joint_angles.items():
            if self.model.existJointName(joint_name):
                joint_id = self.model.getJointId(joint_name)
                # 直接赋值，因为现在所有关节都是 1 维的 (nq=1)
                idx_q = self.model.joints[joint_id].idx_q
                q[idx_q] = angle
            else:
                print(f"[警告] URDF 中找不到关节: '{joint_name}'")

        self.fk_all(q)
        return q


if __name__ == "__main__":
    robot = LW_Leg()
    # talos.load_visualizer()
    q = robot.go_neutral()
    # talos.display(q)

    robot.fk_all(q)

    for i, frame in enumerate(robot.model.frames):
        print(
            f"Frame {i}: Name = {frame.name}, Type = {frame.type}, Parrent = {frame.parentJoint} "
        )

    print(robot.model)

    # for i in range(talos.model.fr):
    # print(i, talos.data.joints[i].M)
