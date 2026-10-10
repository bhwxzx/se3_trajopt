"""Shared saved-node and terminal-geometry helpers for LW validation tools."""

import json

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    raise TypeError(type(value).__name__)

def write_json(path, value):
    path.write_text(json.dumps(value, default=serializable, indent=2) + "\n")

def pack_saved_nodes(problem, saved):
    from nltrajopt.utils import pin2rep
    nodes = saved["solution"]["nodes"]
    if len(nodes) != problem.K:
        raise ValueError("Baseline node count differs from the current contact schedule")
    w = problem.x0.copy()
    for nd, state in zip(problem.nodes, nodes):
        if set(state["forces"]) != set(nd.contact_phase_fnames):
            raise ValueError(f"Contact schedule mismatch at node {nd.k}")
        q = np.asarray(state["q"])
        w[nd.dt_id] = state["dt"]
        w[nd.q_id] = pin2rep(q)
        w[nd.vq_id] = state["v"]
        w[nd.aq_id] = state["a"]
        if hasattr(nd,'stance_end_a_id'):
            w[nd.stance_end_a_id]=state.get('stance_end_a',state['a'])
            for name,ids in nd.stance_end_forces_ids.items():w[ids]=state.get('stance_end_forces',state['forces'])[name]
        if hasattr(nd, 'rolling_projection_id'):
            w[nd.rolling_projection_id] = state.get('rolling_projection', np.zeros(6))
        if hasattr(nd,'midpoint_forces_ids'):
            if saved.get('info',{}).get('motion_config',{}).get('wheel_curve_consistency') and set(state.get('midpoint_forces',{}))!=set(nd.midpoint_forces_ids):
                raise ValueError('Saved curve-control solution has missing midpoint forces')
            for name,ids in nd.midpoint_forces_ids.items():
                w[ids]=state.get('midpoint_forces',{}).get(name,.5*(np.array(state['forces'][name])+np.array(nodes[nd.k+1]['forces'][name])))
        pin.forwardKinematics(problem.model, problem.data, q)
        pin.updateFramePlacements(problem.model, problem.data)
        for frame in nd.contact_phase_fnames:
            w[nd.forces_ids[frame]] = state["forces"][frame]
            # Old JSON files omit these optimization variables; reconstruct FK.
            w[nd.contact_pos_ids[frame]] = state.get("contact_positions", {}).get(
                frame, problem.data.oMf[problem.model.getFrameId(frame)].translation)
    return w


def terminal_target(model, wheels, reference):
    """Small height/pitch correction; wheel contact x must support COM x."""
    data = model.createData()
    rpy = Rotation.from_quat(reference[3:7]).as_euler("xyz")

    def configuration(x):
        q = reference.copy()
        q[2] += x[0]
        q[3:7] = Rotation.from_euler("xyz", rpy + [0., x[1], 0.]).as_quat()
        return q

    def residual(x):
        q = configuration(x)
        state = wheels.evaluate(q)
        com = pin.centerOfMass(model, data, q).copy()
        p = np.array([item["position"] for item in state.values()])
        return np.array([p[:, 2].mean(), com[0] - p[:, 0].mean()])

    # Two equations with a bounded final correction and independently checked
    # residuals; no change to the original joint target angles is needed.
    x = np.zeros(2)
    for _ in range(20):
        value = residual(x)
        if np.max(abs(value)) < 1e-12:
            break
        eps = 1e-6
        derivative = np.column_stack([(residual(x + eps * axis) - residual(x - eps * axis)) / (2 * eps)
                                      for axis in np.eye(2)])
        x -= np.linalg.solve(derivative, value)
    if np.max(np.abs(residual(x))) > 1e-9 or np.max(np.abs(x)) > .02:
        raise RuntimeError(f"Terminal correction failed or exceeded 2 cm / 0.02 rad: {x}")
    q = configuration(x)
    state = wheels.evaluate(q)
    com = pin.centerOfMass(model, data, q).copy()
    left, right = [state[name]["position"] for name in wheels.names]
    fraction = (com[1] - right[1]) / (left[1] - right[1])
    mg = pin.computeTotalMass(model) * np.linalg.norm(model.gravity.linear)
    forces_world = {wheels.names[0]: np.array([0., 0., fraction * mg]),
                    wheels.names[1]: np.array([0., 0., (1 - fraction) * mg])}
    torque = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv)).copy()
    for name in wheels.names:
        torque -= state[name]["Jmaterial"].T @ forces_world[name]
    if np.max(np.abs(torque[:6])) > 1e-6 or np.any(abs(torque[6:]) > model.effortLimit[6:]):
        raise RuntimeError("New terminal target is not statically feasible")
    return q, {"base_height_delta_m": x[0], "base_pitch_delta_rad": x[1],
               "q": q, "contact_positions": {n: s["position"] for n, s in state.items()},
               "forces": {n: state[n]["force_rotation"].T @ f for n, f in forces_world.items()},
               "generalized_static_torque": torque, "max_base_residual": np.max(abs(torque[:6]))}
