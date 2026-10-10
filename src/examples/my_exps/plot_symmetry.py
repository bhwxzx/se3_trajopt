import os
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区 =================
# 替换为你实际生成的 CSV 文件名（确保与该脚本在同级目录，或写绝对路径）
CSV_FILENAME = "wheel_to_leg_transform_60hz.csv" 
FPS = 60.0
# ==========================================

def analyze_symmetry():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(script_dir, CSV_FILENAME)

    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        print("请检查文件名是否正确，或修改 CSV_FILENAME 变量。")
        return

    # 读取纯数据 CSV
    print(f"📂 正在加载数据: {CSV_FILENAME}")
    data = np.loadtxt(csv_path, delimiter=",")
    num_frames = data.shape[0]
    
    # 构建时间轴
    time_axis = np.arange(num_frames) / FPS

    # 定义左右关节的索引配对 (Right_idx, Left_idx, 关节名称)
    # 根据之前定义的格式：7,8:Hip | 9,10:Thigh | 11,12:Shank | 13,14:Foot | 15,16:Wheel
    joint_pairs =[
        (7, 8, "Hip Joint"),
        (9, 10, "Thigh Joint"),
        (11, 12, "Shank Joint"),
        (13, 14, "Foot Joint"),
        (15, 16, "Wheel Joint")
    ]

    # 创建一个 5行 x 2列 的画布
    # 左列：左右关节运动轨迹对比 | 右列：左右误差 (Right - Left)
    fig, axes = plt.subplots(nrows=len(joint_pairs), ncols=2, figsize=(14, 12), sharex=True)
    fig.suptitle("Left-Right Joint Symmetry Analysis (60Hz Trajectory)", fontsize=16, fontweight='bold')

    print("\n📊 对称性量化分析 (Right vs Left):")
    print("-" * 50)

    for i, (r_idx, l_idx, joint_name) in enumerate(joint_pairs):
        right_traj = data[:, r_idx]
        left_traj = data[:, l_idx]
        
        # 计算误差
        # 注意：如果 URDF 中左右关节是镜像对称(正负相反)定义的，请在此处改为 right_traj + left_traj
        error_traj = right_traj + left_traj 
        max_error = np.max(np.abs(error_traj))
        
        print(f"🔹 {joint_name:<15} | 最大偏差: {max_error:.6f} rad")

        # ================= 左列：绝对轨迹对比 =================
        ax_traj = axes[i, 0]
        # 右侧关节用红色实线，左侧关节用蓝色虚线（如果完美对称，蓝色虚线会严丝合缝地盖在红色实线上）
        ax_traj.plot(time_axis, right_traj, color='red', linewidth=2, label='Right')
        ax_traj.plot(time_axis, left_traj, color='dodgerblue', linestyle='--', linewidth=2, label='Left')
        
        ax_traj.set_ylabel("Angle (rad)", fontsize=10)
        ax_traj.set_title(f"{joint_name} Trajectory", fontsize=12)
        ax_traj.grid(True, linestyle=':', alpha=0.7)
        if i == 0:
            ax_traj.legend(loc="upper right")

        # ================= 右列：误差曲线 =================
        ax_err = axes[i, 1]
        ax_err.plot(time_axis, error_traj, color='purple', linewidth=1.5)
        
        # 填充误差区域以便于直观观察
        ax_err.fill_between(time_axis, error_traj, 0, color='purple', alpha=0.2)
        
        ax_err.set_ylabel("Error (rad)", fontsize=10)
        ax_err.set_title(f"{joint_name} Asymmetry (Right - Left)", fontsize=12)
        ax_err.grid(True, linestyle=':', alpha=0.7)

    # 底部加上 X 轴标签
    axes[-1, 0].set_xlabel("Time (seconds)", fontsize=12)
    axes[-1, 1].set_xlabel("Time (seconds)", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 给主标题留出空间
    
    # # 保存图片并展示
    # save_fig_path = os.path.join(script_dir, "symmetry_analysis_plot.png")
    # plt.savefig(save_fig_path, dpi=300)
    # print("-" * 50)
    # print(f"✅ 对称性分析图表已保存至: {save_fig_path}")
    
    plt.show()

if __name__ == "__main__":
    analyze_symmetry()