"""Flat-ground landing costs in world axes, with analytic SE(3) derivatives."""

import numpy as np
import pinocchio as pin

from nltrajopt.se3tangent import q_tan2pin


class ContactNormalApproachCost:
    """0.5 * weight * sum(min(world contact vz, 0)^2).

    This penalizes downward point velocity, including articulated motion, and
    does not constrain tangential wheel rolling. It is a fixed-node event cost.
    """

    def __init__(self, model, frames, weight):
        self.model = model
        self.data = model.createData()
        self.frame_ids = [model.getFrameId(name) for name in frames]
        self.weight = float(weight)
        self.axis = 2
        self.negative_only = True

    def _residual(self, velocity):
        value = float(velocity[self.axis])
        return min(value, 0.) if self.negative_only else value

    def obj(self, w, node, next_node=None):
        q = q_tan2pin(w[node.q_id])
        pin.forwardKinematics(self.model, self.data, q, w[node.vq_id])
        return 0.5 * self.weight * sum(
            self._residual(pin.getFrameVelocity(self.model, self.data, fid,
                                               pin.LOCAL_WORLD_ALIGNED).linear) ** 2
            for fid in self.frame_ids)

    def grad(self, w, gradient, node, next_node=None):
        q_rep = w[node.q_id]
        q = q_tan2pin(q_rep)
        pin.computeForwardKinematicsDerivatives(
            self.model, self.data, q, w[node.vq_id], np.zeros(self.model.nv))
        pin.updateFramePlacements(self.model, self.data)
        for fid in self.frame_ids:
            rotation = self.data.oMf[fid].rotation
            local_velocity = pin.getFrameVelocity(self.model, self.data, fid, pin.LOCAL).linear
            residual = self._residual(rotation @ local_velocity)
            if residual == 0:
                continue
            dq, dv = pin.getFrameVelocityDerivatives(
                self.model, self.data, fid, pin.LOCAL)
            # Differentiate the actual Euclidean point velocity R * v_local.
            # Include dR/dq explicitly; a spatial-motion derivative expressed
            # in LWA is not itself this Euclidean derivative.
            dq = (rotation @ dq[:3] - rotation @ pin.skew(local_velocity) @ dv[3:])[self.axis].copy()
            dv = (rotation @ dv[:3])[self.axis].copy()
            dq[:6] = dq[:6] @ pin.Jexp6(q_rep[:6])
            gradient[node.q_id] += self.weight * residual * dq
            gradient[node.vq_id] += self.weight * residual * dv


class ContactLateralVelocityCost(ContactNormalApproachCost):
    """Penalize world-y motion for this straight, flat-ground maneuver.

    Keeps world-x free, avoiding a no-slip assumption for forward wheel motion.
    """

    def __init__(self, model, frames, weight):
        super().__init__(model, frames, weight)
        self.axis = 1
        self.negative_only = False


class WheelMaterialVelocityCost:
    """Landing velocity cost using the same wheel contact as the constraints."""
    def __init__(self, wheels, weight, axis=2, negative_only=True):
        self.wheels, self.weight = wheels, float(weight)
        self.axis, self.negative_only = axis, negative_only

    def obj(self, w, node, next_node=None):
        state = self.wheels.evaluate(q_tan2pin(w[node.q_id]), w[node.vq_id])
        values = [item["velocity"][self.axis] for item in state.values()]
        if self.negative_only:
            values = np.minimum(values, 0.)
        return .5 * self.weight * np.dot(values, values)

    def grad(self, w, gradient, node, next_node=None):
        q_rep = w[node.q_id]
        state = self.wheels.evaluate(q_tan2pin(q_rep), w[node.vq_id], derivatives=True)
        for item in state.values():
            value = item["velocity"][self.axis]
            residual = min(value, 0.) if self.negative_only else value
            dq = item["velocity_dq"][self.axis].copy()
            dq[:6] = dq[:6] @ pin.Jexp6(q_rep[:6])
            gradient[node.q_id] += self.weight * residual * dq
            gradient[node.vq_id] += self.weight * residual * item["Jmaterial"][self.axis]


class TotalNormalOverloadCost:
    """0.5 * weight * max(sum(world Fz) / mg - 1, 0)^2.

    A soft overload penalty during early wheel support; leaves hard force limits
    unchanged. Uses local saved force variables rotated with current kinematics.
    """

    def __init__(self, model, weight):
        self.model = model
        self.data = model.createData()
        self.weight = float(weight)
        self.body_weight = float(pin.computeTotalMass(model) * np.linalg.norm(model.gravity.linear))

    def _residual(self, w, node):
        pin.forwardKinematics(self.model, self.data, q_tan2pin(w[node.q_id]))
        pin.updateFramePlacements(self.model, self.data)
        total = sum((self.data.oMf[self.model.getFrameId(name)].rotation @ w[node.forces_ids[name]])[2]
                    for name in node.contact_phase_fnames)
        return max(float(total / self.body_weight - 1), 0.)

    def obj(self, w, node, next_node=None):
        return 0.5 * self.weight * self._residual(w, node) ** 2

    def grad(self, w, gradient, node, next_node=None):
        residual = self._residual(w, node)
        if residual == 0:
            return
        q_rep = w[node.q_id]
        q = q_tan2pin(q_rep)
        dq = np.zeros(self.model.nv)
        factor = self.weight * residual / self.body_weight
        for name in node.contact_phase_fnames:
            fid = self.model.getFrameId(name)
            rotation = self.data.oMf[fid].rotation.copy()
            jac = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.LOCAL)
            dq -= (rotation @ pin.skew(w[node.forces_ids[name]]) @ jac[3:])[2]
            gradient[node.forces_ids[name]] += factor * rotation[2]
        dq[:6] = dq[:6] @ pin.Jexp6(q_rep[:6])
        gradient[node.q_id] += factor * dq


def add_landing_costs(problem, velocity_weight=0., force_weight=0., lateral_weight=0., force_all_stance=False):
    """Fixed 50 ms experiment: ramp approach cost over the last 100 ms.

    Force overload cost spans the first 200 ms, or all wheel contact if requested. No mesh or
    contact changes are made. Returns exact nodes and weights for provenance.
    """
    wheels = ["left_wheel_contact_link", "right_wheel_contact_link"]
    landing = next(k for k, node in enumerate(problem.nodes)
                   if all(name in node.contact_phase_fnames for name in wheels))
    records = []
    if velocity_weight:
        for k, factor in ((landing - 2, .25), (landing - 1, .5), (landing, 1.)):
            weight = float(velocity_weight * factor)
            cost = (WheelMaterialVelocityCost(problem.wheel_contact_model, weight)
                    if hasattr(problem, "wheel_contact_model") else ContactNormalApproachCost(problem.model, wheels, weight))
            problem.nodes[k].costs_list.append(cost)
            records.append({"node": k, "kind": "normal_approach", "weight": weight})
    if force_weight:
        stop = problem.K if force_all_stance else min(landing + 5, problem.K)
        for k in range(landing, stop):
            problem.nodes[k].costs_list.append(TotalNormalOverloadCost(problem.model, force_weight))
            records.append({"node": k, "kind": "normal_overload", "weight": float(force_weight)})
    if lateral_weight:
        for k, factor in ((landing - 2, .25), (landing - 1, .5), (landing, 1.)):
            weight = float(lateral_weight * factor)
            cost = (WheelMaterialVelocityCost(problem.wheel_contact_model, weight, axis=1, negative_only=False)
                    if hasattr(problem, "wheel_contact_model") else ContactLateralVelocityCost(problem.model, wheels, weight))
            problem.nodes[k].costs_list.append(cost)
            records.append({"node": k, "kind": "lateral_speed", "weight": weight})
    return {"velocity_weight": float(velocity_weight), "force_weight": float(force_weight),
            "lateral_weight": float(lateral_weight),
            "force_all_stance": bool(force_all_stance),
            "landing_node": landing, "terms": records}
