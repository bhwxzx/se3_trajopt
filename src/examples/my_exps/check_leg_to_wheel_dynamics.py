"""Inspect saved optimizer nodes; position-only CSVs do not contain contact loads.

Run with the repository's se3_trajopt environment, for example:
  python src/examples/my_exps/check_leg_to_wheel_dynamics.py SOLUTION.json

Forces in saved nodes are expressed in each contact frame. Torques follow the
same external-wrench convention as WholeBodyDynamics, including moment arms.
The reported forces are discrete optimizer loads, not collision impulse peaks.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[3]
URDF = ROOT / "src/robots/LW_Leg/LW_Leg/urdf/LW.urdf"
JOINT_NAMES = [
    f"{side}_{part}_joint"
    for part in ("hip", "thigh", "shank", "foot", "wheel")
    for side in ("right", "left")
]


def csv_error(nodes, model, csv_path):
    """Check direct 60 Hz node export or the legacy interpolated/trimmed export."""
    q = np.array([n["q"] for n in nodes])
    t = np.r_[0.0, np.cumsum([n["dt"] for n in nodes][:-1])]
    tt = np.arange(0, t[-1], 1 / 60)
    if not len(tt) or tt[-1] < t[-1]:
        tt = np.append(tt, t[-1])
    ids = [model.joints[model.getJointId(name)].idx_q for name in JOINT_NAMES]
    actual = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    if actual.shape == (len(nodes), 17) and np.allclose(np.diff(t), 1 / 60, rtol=0, atol=1e-9):
        exported = np.column_stack((q[:, :7], q[:, ids]))
    else:
        exported = np.column_stack((
            PchipInterpolator(t, q[:, :3])(tt),
            Slerp(t, Rotation.from_quat(q[:, 3:7]))(tt).as_quat(),
            PchipInterpolator(t, q[:, ids])(tt),
        ))[1:-1]
    if actual.shape != exported.shape:
        return {"matches": False, "reason": "shape mismatch"}
    error = float(np.max(np.abs(actual - exported)))
    return {"matches": error <= 5.01e-7, "max_abs_error": error}


def analyze(saved, model):
    nodes = saved["solution"]["nodes"]
    wheels = None
    if "wheel_contact_model" in saved["info"]:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from nltrajopt.wheel_contact import WheelContactModel
        wheels = WheelContactModel.from_metadata(model, saved["info"]["wheel_contact_model"])
    data = model.createData()
    t = np.r_[0.0, np.cumsum([n["dt"] for n in nodes][:-1])]
    frames = sorted({name for n in nodes for name in n["forces"]})
    wheel_frames = [name for name in frames if "wheel" in name]
    if not wheel_frames:
        raise ValueError("No wheel contacts found in this solution")
    force = np.zeros((len(nodes), len(frames), 3))
    tau = np.zeros((len(nodes), model.nv))
    velocity = np.zeros_like(force)
    height = np.zeros((len(nodes), len(frames)))
    com_position = np.zeros((len(nodes), 3))
    com_velocity = np.zeros_like(com_position)
    jacobian_error = 0.0
    for k, node in enumerate(nodes):
        q, v, a = (np.asarray(node[key]) for key in ("q", "v", "a"))
        pin.forwardKinematics(model, data, q, v, a)
        pin.updateFramePlacements(model, data)
        wheel_state = wheels.evaluate(q, v) if wheels else {}
        external = [pin.Force.Zero() for _ in range(model.njoints)]
        for j, name in enumerate(frames):
            fid = model.getFrameId(name)
            frame = model.frames[fid]
            height[k, j] = data.oMf[fid].translation[2]
            velocity[k, j] = pin.getFrameVelocity(
                model, data, fid, pin.LOCAL_WORLD_ALIGNED
            ).linear.copy()
            if name in wheel_state:
                height[k, j] = wheel_state[name]["position"][2]
                velocity[k, j] = wheel_state[name]["velocity"]
            if name in node["forces"]:
                local_force = np.asarray(node["forces"][name])
                force[k, j] = data.oMf[fid].rotation @ local_force
                if name in wheel_state:
                    item = wheel_state[name]
                    frame = model.frames[wheels.frames[name]]
                    local_force = item["rotation"].T @ force[k, j]
                    local_offset = item["rotation"].T @ item["offset"]
                    wrench = pin.Force(local_force, np.cross(local_offset, local_force))
                else:
                    wrench = pin.Force(local_force, np.zeros(3))
                external[frame.parentJoint] += frame.placement.act(wrench)
        tau[k] = pin.rnea(model, data, q, v, a, external).copy()
        # Independent check: inverse dynamics without forces minus sum(J^T f).
        check = pin.rnea(model, data, q, v, a).copy()
        for name, local_force in node["forces"].items():
            if name in wheel_state:
                item = wheel_state[name]
                check -= item["Jmaterial"].T @ item["force_rotation"] @ np.asarray(local_force)
                continue
            jac = pin.computeFrameJacobian(
                model, data, q, model.getFrameId(name), pin.LOCAL
            )
            check -= jac[:3].T @ np.asarray(local_force)
        jacobian_error = max(jacobian_error, float(np.max(np.abs(check - tau[k]))))
        pin.centerOfMass(model, data, q, v)
        com_position[k] = data.com[0]
        com_velocity[k] = data.vcom[0]

    landing = next(k for k, n in enumerate(nodes) if any(f in n["forces"] for f in wheel_frames))
    flight = next(k for k, n in enumerate(nodes[:landing]) if not n["forces"])
    indices = [model.joints[model.getJointId(n)].idx_v for n in JOINT_NAMES]
    joints = []
    for name, j in zip(JOINT_NAMES, indices):
        k = int(np.argmax(np.abs(tau[:, j])))
        kl = landing + int(np.argmax(np.abs(tau[landing:, j])))
        joints.append({
            "joint": name, "peak_abs_Nm": float(abs(tau[k, j])),
            "signed_peak_Nm": float(tau[k, j]), "peak_time_s": float(t[k]),
            "landing_peak_abs_Nm": float(abs(tau[kl, j])),
            "landing_peak_time_s": float(t[kl]),
            "limit_Nm": float(model.effortLimit[j]),
            "limit_fraction": float(abs(tau[k, j]) / model.effortLimit[j]),
        })
    total_force = force.sum(axis=1)
    kp = landing + int(np.argmax(total_force[landing:, 2]))
    mass = float(pin.computeTotalMass(model))
    weight = mass * np.linalg.norm(model.gravity.linear)
    dt = np.array([n["dt"] for n in nodes])
    v = np.array([n["v"] for n in nodes])
    a = np.array([n["a"] for n in nodes])
    report = {
        "solver_info": saved["info"], "mass_kg": mass, "weight_N": float(weight),
        "node_count": len(nodes), "sample_dt_s": float(dt[0]),
        "flight_start_s": float(t[flight]), "landing_start_s": float(t[landing]),
        "flight_duration_s": float(t[landing] - t[flight]),
        "last_node_time_s": float(t[-1]), "joint_torques": joints,
        "landing_total_Fz_peak_N": float(total_force[kp, 2]),
        "landing_total_Fz_peak_time_s": float(t[kp]),
        "landing_peak_bodyweights": float(total_force[kp, 2] / weight),
        "default_per_contact_force_limit_N": float(1.5 * weight),
        "wheel_contacts": {},
        "com_touchdown_velocity_world_m_s": com_velocity[landing].tolist(),
        "landing_window": [
            {"node": k, "time_s": float(t[k]), "total_Fz_N": float(total_force[k, 2]),
             "com_z_m": float(com_position[k, 2]), "com_vz_m_s": float(com_velocity[k, 2]),
             "wheel_vz_m_s": {name: float(velocity[k, frames.index(name), 2]) for name in wheel_frames},
             "wheel_height_m": {name: float(height[k, frames.index(name)]) for name in wheel_frames}}
            for k in range(len(nodes)) if t[landing] - .100001 <= t[k] <= t[landing] + .200001],
        "base_force_residual_max_abs_N": float(np.max(np.abs(tau[:, :3]))),
        "base_moment_residual_max_abs_Nm": float(np.max(np.abs(tau[:, 3:6]))),
        "velocity_integration_max_abs": float(np.max(np.abs(v[1:] - v[:-1] - a[:-1] * dt[:-1, None]))),
        "rnea_vs_jacobian_max_abs": jacobian_error,
        "limitations": [
            "Uses current URDF; saved solutions do not embed the original model.",
            "Loads are reconstructed at optimizer nodes, not from CSV derivatives.",
            "No collision impulse/compliance model; these are not physical impact peaks.",
            "Solver status must be checked before treating this as a feasible trajectory.",
        ],
    }
    for name in wheel_frames:
        j = frames.index(name)
        report["wheel_contacts"][name] = {
            "peak_Fz_N": float(force[landing:, j, 2].max()),
            "peak_force_norm_N": float(np.linalg.norm(force[landing:, j], axis=1).max()),
            "touchdown_force_world_N": force[landing, j].tolist(),
            "touchdown_velocity_world_m_s": velocity[landing, j].tolist(),
            "touchdown_height_m": float(height[landing, j]),
            "last_airborne_vz_m_s": float(velocity[landing - 1, j, 2]),
            "min_vz_last_100ms_m_s": float(velocity[max(0, landing - 2):landing + 1, j, 2].min()),
        }
    return report, t, tau[:, indices], force, frames


def save_outputs(output, report, t, tau, force, frames):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ["time_s"] + [n + "_Nm" for n in JOINT_NAMES]
    columns += [f"{name}_F{axis}_world_N" for name in frames for axis in "xyz"]
    np.savetxt(output / "node_loads.csv", np.column_stack((t, tau, force.reshape(len(t), -1))),
               delimiter=",", header=",".join(columns), comments="")
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for i, part in enumerate(("hip", "thigh", "shank", "foot", "wheel")):
        ax = axes.flat[i]
        for side_idx, side in enumerate(("right", "left")):
            j = 2 * i + side_idx
            ax.plot(t, tau[:, j], label=side)
        limit = report["joint_torques"][2 * i]["limit_Nm"]
        ax.axhline(limit, color="red", linestyle=":", label="URDF limit")
        ax.axhline(-limit, color="red", linestyle=":")
        ax.set_title(part.capitalize() + " joint torque")
        ax.set_ylabel("N m")
        ax.legend(fontsize=8)
    ax = axes.flat[-1]
    ax.plot(t, force.sum(axis=1)[:, 2], label="all contacts: total Fz")
    for j, name in enumerate(frames):
        if "wheel" in name:
            ax.plot(t, force[:, j, 2], label=name.split("_")[0] + " wheel Fz")
    ax.axhline(report["weight_N"], color="gray", linestyle=":", label="body weight")
    ax.set_title("World vertical contact force (discrete optimizer nodes)")
    ax.set_ylabel("N")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.axvspan(report["flight_start_s"], report["landing_start_s"], color="gray", alpha=0.15)
        ax.axvline(report["landing_start_s"], color="black", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"Leg to wheel loads | solver status {report['solver_info']['solved']} | not collision peaks")
    fig.tight_layout()
    fig.savefig(output / "loads.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("leg_to_wheel_transform_60hz.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    saved = json.loads(args.solution.read_text())
    model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    report, t, tau, force, frames = analyze(saved, model)
    report["source_solution"] = str(args.solution.resolve())
    report["source_solution_sha256"] = hashlib.sha256(args.solution.read_bytes()).hexdigest()
    report["urdf_path"] = str(URDF)
    report["urdf_sha256"] = hashlib.sha256(URDF.read_bytes()).hexdigest()
    report["csv_comparison"] = csv_error(saved["solution"]["nodes"], model, args.csv)
    output = args.output or args.solution.parent / (args.solution.stem + "_dynamics")
    save_outputs(output, report, t, tau, force, frames)
    print(json.dumps(report, indent=2))
    print(f"Saved analysis to {output.resolve()}")


if __name__ == "__main__":
    main()
