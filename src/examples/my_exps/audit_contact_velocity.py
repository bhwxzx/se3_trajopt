"""Audit saved LW contact kinematics without changing or solving the trajectory.

All point positions/velocities are in the world frame, with flat ground z=0.
Uses optimizer v, manifold directional derivatives, node position secants,
and finite differences of FK of the actual exported CSV as separate signals.
Wheel collision STL vertices are checked independently of the contact proxies.

Run in the se3_trajopt environment:
  python src/examples/my_exps/audit_contact_velocity.py --solution SOLUTION.json \
      --csv TRAJECTORY.csv --output NEW_DIRECTORY
CSV optimizer times default to csv_optimizer_times_s.csv beside the CSV.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from check_leg_to_wheel_dynamics import ROOT, URDF, JOINT_NAMES
from visualize_trajectory_csv import load_configurations


SIDES = ("left", "right")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, default=convert, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vector_columns(row, prefix, vector, unit):
    row.update({f"{prefix}_{axis}_{unit}": float(value) for axis, value in zip("xyz", vector)})


def wheel_geometry(urdf):
    """Read collision meshes and apply URDF mesh scale/origin into wheel frame."""
    xml = ET.parse(urdf).getroot()
    vertices, metadata, paths = {}, {}, []
    for side in SIDES:
        collision = xml.find(f"link[@name='{side}_wheel_link']/collision")
        mesh = collision.find("geometry/mesh")
        path = urdf.parent.parent / mesh.attrib["filename"]
        raw = path.read_bytes()
        count = struct.unpack_from("<I", raw, 80)[0]
        if len(raw) != 84 + 50 * count:
            raise ValueError(f"Expected binary STL with 50-byte triangles: {path}")
        dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        points = np.unique(np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].reshape(-1, 3), axis=0).astype(float)
        scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        origin = collision.find("origin")
        translation = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        points = (points * scale) @ pin.rpy.rpyToMatrix(rpy).T + translation
        vertices[side] = points
        joint = xml.find(f"joint[@name='{side}_wheel_joint']")
        proxy_joint = xml.find(f"joint[@name='{side}_wheel_contact_joint']")
        center_local = np.fromstring(joint.find("origin").get("xyz"), sep=" ")
        proxy_local = np.fromstring(proxy_joint.find("origin").get("xyz"), sep=" ")
        metadata[side] = {
            "collision_mesh": str(path.resolve()), "sha256": digest(path),
            "unique_vertices": len(points), "wheel_frame_bounds_m": [points.min(axis=0), points.max(axis=0)],
            "max_radius_about_local_y_m": np.linalg.norm(points[:, [0, 2]], axis=1).max(),
            "wheel_parent": joint.find("parent").get("link"),
            "proxy_parent": proxy_joint.find("parent").get("link"),
            "proxy_offset_from_axle_in_parent_m": proxy_local - center_local,
            "proxy_offset_norm_m": np.linalg.norm(proxy_local - center_local),
            "wheel_joint_axis": np.fromstring(joint.find("axis").get("xyz"), sep=" "),
        }
        paths.append(path)
    return vertices, metadata, paths


def sample(model, data, q, frames, vertices, v=None):
    if v is None:
        pin.forwardKinematics(model, data, q)
    else:
        pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)
    positions = np.array([data.oMf[model.getFrameId(name)].translation.copy() for name in frames])
    velocities = None if v is None else np.array([
        pin.getFrameVelocity(model, data, model.getFrameId(name), pin.LOCAL_WORLD_ALIGNED).linear.copy()
        for name in frames
    ])
    geometry = {}
    for side in SIDES:
        fid = model.getFrameId(f"{side}_wheel_link")
        pose = data.oMf[fid]
        points = vertices[side]
        lowest = int(np.argmin(points @ pose.rotation[2, :] + pose.translation[2]))
        offset_world = pose.rotation @ points[lowest]
        position = pose.translation + offset_world
        values = {"mesh_min_z_m": float(position[2]), "center_z_m": float(pose.translation[2]),
                  "lowest_vertex_index": lowest}
        if v is not None:
            motion = pin.getFrameVelocity(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
            # Instantaneous MATERIAL velocity of the selected mesh vertex.
            # This is not the finite difference of a changing argmin vertex.
            material_velocity = motion.linear + np.cross(motion.angular, offset_world)
            vector_columns(values, "lowest_material_v", material_velocity, "m_s")
        geometry[side] = values
    return positions, velocities, geometry


def audit(args):
    saved = json.loads(args.solution.read_text())
    if "wheel_contact_model" in saved["info"]:
        raise ValueError("This audit targets legacy fixed proxies; use the corrected run's contact_validation.json and wheel_contacts.csv")
    nodes = saved["solution"]["nodes"]
    if len(nodes) < 3:
        raise ValueError("Need at least three optimizer nodes")
    q = np.asarray([n["q"] for n in nodes])
    v = np.asarray([n["v"] for n in nodes])
    a = np.asarray([n["a"] for n in nodes])
    dt = np.asarray([n["dt"] for n in nodes])
    if np.any(dt <= 0) or not all(np.isfinite(x).all() for x in (q, v, a, dt)):
        raise ValueError("Invalid optimizer state or time steps")
    t = np.r_[0., np.cumsum(dt[:-1])]
    model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    data, derivative_data = model.createData(), model.createData()
    vertices, geometry_info, mesh_paths = wheel_geometry(URDF)
    frames = sorted({frame for node in nodes for frame in node["forces"]})
    wheel_ids = [frames.index(f"{side}_wheel_contact_link") for side in SIDES]
    active = np.array([[name in node["forces"] for name in frames] for node in nodes])
    landing = int(np.flatnonzero(active[:, wheel_ids].any(axis=1))[0])

    times_path = args.times or args.csv.with_name("csv_optimizer_times_s.csv")
    csv_t = np.loadtxt(times_path, delimiter=",", skiprows=1, ndmin=1)
    csv_q = load_configurations(args.csv, model)
    if len(csv_t) != len(csv_q) or len(csv_t) < 3 or np.any(np.diff(csv_t) <= 0) or not np.isfinite(csv_t).all():
        raise ValueError("CSV timestamps must be finite, increasing, and match CSV rows")
    if csv_t[0] < t[0] or csv_t[-1] > t[-1]:
        raise ValueError("CSV timestamps outside optimizer time range")

    # Reproduce the exporter at the provided timestamps to establish provenance.
    columns = [model.joints[model.getJointId(name)].idx_q for name in JOINT_NAMES]
    expected = np.column_stack((PchipInterpolator(t, q[:, :3])(csv_t),
                               Slerp(t, Rotation.from_quat(q[:, 3:7]))(csv_t).as_quat(),
                               PchipInterpolator(t, q[:, columns])(csv_t)))
    actual = np.loadtxt(args.csv, delimiter=",", ndmin=2)
    export_error = float(np.max(np.abs(expected - actual)))
    if export_error > 5.1e-7:
        raise ValueError(f"CSV does not reproduce solution's PCHIP/Slerp export: {export_error}")

    protected = [args.solution, args.csv, times_path, URDF, *mesh_paths,
                 ROOT / "src/examples/my_exps/leg_to_wheel_transform_60hz.csv"]
    source_files = [Path(__file__), Path(__file__).with_name("visualize_trajectory_csv.py"),
                    ROOT / "src/nltrajopt/constraint_models/terrain_constraints.py",
                    ROOT / "src/nltrajopt/constraint_models/semi_euler_integration.py"]
    hashes_before = {str(p.resolve()): digest(p) for p in protected + source_files}
    positions, velocities, geometry_nodes = [], [], []
    validation = {"Jv_vs_frame_velocity_max_abs_m_s": 0., "Jv_vs_manifold_FD_max_abs_m_s": 0.,
                  "proxy_wheel_spin_jacobian_max_abs": 0., "csv_export_max_abs_error": export_error}
    eps = 1e-6
    for k in range(len(nodes)):
        p, vel, geo = sample(model, data, q[k], frames, vertices, v[k])
        positions.append(p)
        velocities.append(vel)
        geometry_nodes.append(geo)
        # Directional difference on the configuration manifold validates both
        # base velocity convention and world-coordinate point velocities.
        derivatives = []
        for sign in (-1., 1.):
            pin.forwardKinematics(model, derivative_data, pin.integrate(model, q[k], sign * eps * v[k]))
            pin.updateFramePlacements(model, derivative_data)
            derivatives.append(np.array([derivative_data.oMf[model.getFrameId(name)].translation.copy() for name in frames]))
        fd = (derivatives[1] - derivatives[0]) / (2 * eps)
        for j, name in enumerate(frames):
            jac = pin.computeFrameJacobian(model, data, q[k], model.getFrameId(name), pin.LOCAL_WORLD_ALIGNED)[:3]
            jv = jac @ v[k]
            validation["Jv_vs_frame_velocity_max_abs_m_s"] = max(validation["Jv_vs_frame_velocity_max_abs_m_s"], float(np.max(np.abs(jv - vel[j]))))
            validation["Jv_vs_manifold_FD_max_abs_m_s"] = max(validation["Jv_vs_manifold_FD_max_abs_m_s"], float(np.max(np.abs(jv - fd[j]))))
            if j in wheel_ids:
                side = name.split("_")[0]
                spin = model.joints[model.getJointId(f"{side}_wheel_joint")].idx_v
                validation["proxy_wheel_spin_jacobian_max_abs"] = max(validation["proxy_wheel_spin_jacobian_max_abs"], float(np.max(np.abs(jac[:, spin]))))
    positions, velocities = np.array(positions), np.array(velocities)
    secants = np.diff(positions, axis=0) / np.diff(t)[:, None, None]
    node_fd = np.gradient(positions, t, axis=0, edge_order=2)
    configuration_secants = np.array([pin.difference(model, q[k], q[k + 1]) / dt[k] for k in range(len(nodes) - 1)])
    validation["semi_euler_velocity_residual_max_abs"] = np.max(np.abs(v[1:] - v[:-1] - dt[:-1, None] * a[:-1]))
    validation["semi_euler_position_residual_max_abs"] = max(float(np.max(np.abs(pin.difference(
        model, q[k + 1], pin.integrate(model, q[k], (v[k] + a[k] * dt[k]) * dt[k]))))) for k in range(len(nodes) - 1))
    validation["manifold_secant_vs_v_next_max_abs"] = np.max(np.abs(configuration_secants - v[1:]))
    assert validation["Jv_vs_frame_velocity_max_abs_m_s"] < 1e-9
    assert validation["Jv_vs_manifold_FD_max_abs_m_s"] < 2e-6

    node_rows, interval_rows = [], []
    for k, node in enumerate(nodes):
        for j, name in enumerate(frames):
            row = {"node": k, "time_s": t[k], "frame": name, "active": int(active[k, j])}
            vector_columns(row, "position", positions[k, j], "m")
            vector_columns(row, "optimizer_Jv", velocities[k, j], "m_s")
            vector_columns(row, "node_centered_fd", node_fd[k, j], "m_s")
            node_rows.append(row)
            if k < len(nodes) - 1:
                ir = {"node_start": k, "time_start_s": t[k], "time_end_s": t[k + 1], "frame": name,
                      "active_start": int(active[k, j]), "active_end": int(active[k + 1, j])}
                vector_columns(ir, "position_secant", secants[k, j], "m_s")
                vector_columns(ir, "optimizer_v_start", velocities[k, j], "m_s")
                vector_columns(ir, "optimizer_v_end", velocities[k + 1, j], "m_s")
                interval_rows.append(ir)

    csv_samples = [sample(model, data, config, frames, vertices) for config in csv_q]
    csv_positions = np.array([item[0] for item in csv_samples])
    csv_velocities = np.gradient(csv_positions, csv_t, axis=0, edge_order=2)
    csv_node_ids = np.clip(np.searchsorted(t, csv_t + 1e-9, side="right") - 1, 0, len(t) - 1)
    csv_active = active[csv_node_ids]
    csv_rows, geometry_rows = [], []
    for i, tt in enumerate(csv_t):
        for j, name in enumerate(frames):
            row = {"csv_row": i, "optimizer_time_s": tt, "frame": name, "scheduled_active": int(csv_active[i, j])}
            vector_columns(row, "position", csv_positions[i, j], "m")
            vector_columns(row, "csv_centered_fd", csv_velocities[i, j], "m_s")
            csv_rows.append(row)
    for k in range(len(nodes)):
        for side, j in zip(SIDES, wheel_ids):
            geometry_rows.append({"node": k, "time_s": t[k], "side": side, "active": int(active[k, j]),
                                  "proxy_z_m": positions[k, j, 2], **geometry_nodes[k][side]})
    csv_geometry_rows = [
        {"csv_row": i, "optimizer_time_s": tt, "side": side,
         "scheduled_active": int(csv_active[i, j]), "proxy_z_m": csv_positions[i, j, 2], **csv_samples[i][2][side]}
        for i, tt in enumerate(csv_t) for side, j in zip(SIDES, wheel_ids)
    ]

    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "solution": str(args.solution.resolve()),
               "solver_info": json.loads(args.solution.read_text())["info"], "landing_node": landing,
               "landing_time_s": t[landing], "world_ground_normal": [0, 0, 1], "checks": validation,
               "wheel_geometry": geometry_info, "per_contact": {}, "wheels": {},
               "conventions": {
                   "optimizer_velocity": "Pinocchio LOCAL_WORLD_ALIGNED frame linear velocity = J(q) v; world components at frame origin.",
                   "node_position_secant": "(p[k+1]-p[k])/dt; interval average, not instantaneous velocity.",
                   "centered_fd": "numpy.gradient of world FK positions on actual sample times; second-order one-sided endpoints.",
                   "csv_time": "Uses optimizer-time sidecar; CSV rows contain no time column and omit original endpoints.",
                   "active": "Presence of contact name in optimizer forces map, not inferred from force magnitude.",
                   "mesh": "Minimum world z over URDF collision STL vertices; rigid geometry against z=0, not simulated compliant contact.",
                   "material_velocity": "Velocity of lowest selected mesh vertex as a wheel material point, includes wheel spin; argmin can change.",
               }, "limitations": [
                   "No impact impulse, compliance, or simulator validation; velocities do not determine impact peak force.",
                   "Finite time differences can disagree with instantaneous velocities even for smooth motion; full tables expose interval endpoints.",
                   "Mesh heights cover wheel collision meshes, not every robot collision geometry.",
                   "Current URDF and mesh hashes recorded; original saved JSON alone does not embed model resources.",
               ]}
    for j, name in enumerate(frames):
        mask = active[:, j]
        both = mask[:-1] & mask[1:]
        summary["per_contact"][name] = {
            "active_node_count": np.count_nonzero(mask),
            "max_abs_active_height_m": np.abs(positions[mask, j, 2]).max(),
            "max_abs_active_instantaneous_normal_speed_m_s": np.abs(velocities[mask, j, 2]).max(),
            "max_active_instantaneous_speed_m_s": np.linalg.norm(velocities[mask, j], axis=1).max(),
            "max_same_contact_interval_secant_speed_m_s": np.linalg.norm(secants[both, j], axis=1).max() if both.any() else None,
        }
    for side, j in zip(SIDES, wheel_ids):
        node_mask, csv_mask = active[:, j], csv_active[:, j]
        mesh_z = np.array([geo[side]["mesh_min_z_m"] for geo in geometry_nodes])
        csv_mesh_z = np.array([item[2][side]["mesh_min_z_m"] for item in csv_samples])
        support_ids = np.flatnonzero(node_mask)
        csv_support_ids = np.flatnonzero(csv_mask)
        min_node = support_ids[np.argmin(mesh_z[node_mask])]
        min_csv = csv_support_ids[np.argmin(csv_mesh_z[csv_mask])]
        coincident = int(np.argmin(abs(csv_t - t[landing])))
        csv_touchdown = csv_velocities[coincident, j] if abs(csv_t[coincident] - t[landing]) < 1e-8 else None
        summary["wheels"][side] = {
            "last_airborne_optimizer_velocity_m_s": velocities[landing - 1, j],
            "touchdown_optimizer_velocity_m_s": velocities[landing, j],
            "touchdown_proxy_height_m": positions[landing, j, 2],
            "touchdown_mesh_height_m": mesh_z[landing],
            "touchdown_lowest_material_velocity_m_s": [geometry_nodes[landing][side][f"lowest_material_v_{axis}_m_s"] for axis in "xyz"],
            "arrival_interval_position_secant_m_s": secants[landing - 1, j],
            "first_support_interval_position_secant_m_s": secants[landing, j],
            "csv_centered_fd_at_touchdown_m_s": csv_touchdown,
            "first_support_interval_optimizer_v_end_m_s": velocities[landing + 1, j],
            "minimum_node_mesh_height_m": mesh_z[min_node], "minimum_node_mesh_time_s": t[min_node],
            "minimum_csv_proxy_height_m": csv_positions[csv_mask, j, 2].min(),
            "minimum_csv_mesh_height_m": csv_mesh_z[min_csv], "minimum_csv_mesh_time_s": csv_t[min_csv],
            "support_nodes_mesh_below_minus_1mm": int(np.count_nonzero(mesh_z[node_mask] < -0.001)),
            "landing_window": [
                {"node": k, "time_s": t[k], "active": bool(active[k, j]), "proxy_z_m": positions[k, j, 2],
                 "optimizer_velocity_m_s": velocities[k, j], "mesh_min_z_m": mesh_z[k]}
                for k in range(len(nodes)) if t[landing] - .100001 <= t[k] <= t[landing] + .250001],
        }

    previous_hashes_path = args.solution.parent.parent / "source_hashes.json"
    if previous_hashes_path.exists():
        saved_hash = json.loads(previous_hashes_path.read_text()).get(str(URDF.relative_to(ROOT)))
        summary["checks"]["current_urdf_matches_experiment_record"] = saved_hash == digest(URDF) if saved_hash else None
        summary["checks"]["experiment_source_hashes_file"] = str(previous_hashes_path.resolve())
    assert hashes_before == {p: digest(Path(p)) for p in hashes_before}, "Inputs changed during audit"
    summary["checks"]["inputs_unchanged"] = True
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("node_contacts", node_rows), ("interval_contacts", interval_rows),
                       ("csv_contacts", csv_rows), ("node_wheel_geometry", geometry_rows), ("csv_wheel_geometry", csv_geometry_rows)):
        write_rows(args.output / f"{name}.csv", rows)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "provenance.json", {"sha256": hashes_before, "command": sys.argv,
                                                "python": sys.executable, "pinocchio_version": pin.__version__,
                                                "pinocchio_module": pin.__file__, "fd_epsilon": eps})
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for column, (side, j) in enumerate(zip(SIDES, wheel_ids)):
        axes[0, column].plot(t, positions[:, j, 2] * 1000, "o-", label="node proxy")
        axes[0, column].plot(t, [g[side]["mesh_min_z_m"] * 1000 for g in geometry_nodes], "o-", label="node wheel mesh")
        axes[0, column].plot(csv_t, csv_positions[:, j, 2] * 1000, ".-", label="CSV proxy")
        axes[0, column].plot(csv_t, [s[2][side]["mesh_min_z_m"] * 1000 for s in csv_samples], ".-", label="CSV wheel mesh")
        axes[0, column].set_ylim(-25, 100)
        axes[0, column].set_title(f"{side.capitalize()} wheel")
        axes[0, column].set_ylabel("Height (mm)")
        for row, axis_index, label in ((1, 2, "Vertical speed (m/s)"), (2, 0, "Forward speed (m/s)")):
            axes[row, column].plot(t, velocities[:, j, axis_index], "o-", label="optimizer Jv")
            axes[row, column].plot((t[1:] + t[:-1]) / 2, secants[:, j, axis_index], "s--", label="node position secant")
            axes[row, column].plot(csv_t, csv_velocities[:, j, axis_index], ".-", label="CSV centered FD")
            axes[row, column].set_ylabel(label)
    for ax in axes.flat:
        ax.set_xlim(t[landing] - .15, t[landing] + .4)
        ax.axvline(t[landing], color="black", ls="--", lw=.8)
        ax.axhline(0, color="black", lw=.5)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        ax.set_xlabel("Optimizer time (s)")
    fig.tight_layout()
    fig.savefig(args.output / "contact_velocity_audit.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "checks": validation,
                      "wheels": {s: {k: val for k, val in summary["wheels"][s].items() if k != "landing_window"} for s in SIDES}},
                     default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item(), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--times", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite an audit")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    audit(args)


if __name__ == "__main__":
    main()
