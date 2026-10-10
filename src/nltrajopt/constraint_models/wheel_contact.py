"""Wheel contact kinematics and whole-body dynamics with moving support points."""

from copy import copy

import numpy as np
import pinocchio as pin

from nltrajopt.constraint_models.abstract_constraint import extend_ids_lists
from nltrajopt.constraint_models.terrain_constraints import TerrainGridContactConstraints
from nltrajopt.constraint_models.wb_dynamics import WholeBodyDynamics
from nltrajopt.se3tangent import q_tan2pin


def tangent_columns(matrix, q_rep):
    result = matrix.copy()
    result[:, :6] = result[:, :6] @ pin.Jexp6(q_rep[:6])
    return result


class RollingWheelContactConstraints(TerrainGridContactConstraints):
    """Keep legacy foot constraints; wheel support uses geometric height + Jv.

    The three wheel velocity rows are material no-slip conditions (world xyz).
    Normal velocity is zero; rolling/side-slip rows include wheel spin and do
    not anchor the geometric support point to a fixed world xy position.
    """
    def __init__(self, terrain, wheels):
        if not np.allclose(terrain.grid, 0., atol=1e-12):
            raise ValueError("RollingWheelContactConstraints currently requires flat z=0 terrain")
        super().__init__(terrain)
        self.wheels = wheels

    def feet(self, node):
        if node is None:
            return None
        result = copy(node)
        result.contact_fnames = [n for n in node.contact_fnames if n not in self.wheels.names]
        result.contact_phase_fnames = [n for n in node.contact_phase_fnames if n not in self.wheels.names]
        return result

    def compute_constraints(self, node_curr, node_next, w, c, model, data):
        super().compute_constraints(self.feet(node_curr), self.feet(node_next), w, c, model, data)
        state = self.wheels.evaluate(q_tan2pin(w[node_curr.q_id]), w[node_curr.vq_id])
        for name, item in state.items():
            if name not in node_curr.contact_phase_fnames:
                c[node_curr.c_z_ids[name]] = item["position"][2]
            else:
                c[node_curr.c_contact_kinematics_ids[name]] = item["position"] - w[node_curr.contact_pos_ids[name]]
                c[node_curr.c_z_ids[name]] = w[node_curr.contact_pos_ids[name]][2]
                c[node_curr.c_vel_ids[name]] = item["velocity"]

    def compute_jacobians(self, node_curr, node_next, w, jac, model, data):
        super().compute_jacobians(self.feet(node_curr), self.feet(node_next), w, jac, model, data)
        q_rep = w[node_curr.q_id]
        state = self.wheels.evaluate(q_tan2pin(q_rep), w[node_curr.vq_id], derivatives=True)
        for name, item in state.items():
            Dp = tangent_columns(item["Jgeometry"], q_rep)
            if name not in node_curr.contact_phase_fnames:
                jac[node_curr.c_z_ids[name], node_curr.q_id] = Dp[2]
            else:
                jac[node_curr.c_contact_kinematics_ids[name], node_curr.q_id] = Dp
                jac[node_curr.c_contact_kinematics_ids[name], node_curr.contact_pos_ids[name]] = -np.eye(3)
                jac[node_curr.c_z_ids[name], node_curr.contact_pos_ids[name]] = [0., 0., 1.]
                jac[node_curr.c_vel_ids[name], node_curr.q_id] = tangent_columns(item["velocity_dq"], q_rep)
                jac[node_curr.c_vel_ids[name], node_curr.vq_id] = item["Jmaterial"]

    def get_structure_ids(self, node_curr, node_next, row_ids, col_ids):
        super().get_structure_ids(self.feet(node_curr), self.feet(node_next), row_ids, col_ids)
        for name in self.wheels.names:
            if name not in node_curr.contact_phase_fnames:
                extend_ids_lists(row_ids, col_ids, node_curr.c_z_ids[name], node_curr.q_id)
            else:
                for rows, cols in ((node_curr.c_contact_kinematics_ids[name], node_curr.q_id),
                                   (node_curr.c_contact_kinematics_ids[name], node_curr.contact_pos_ids[name]),
                                   (node_curr.c_z_ids[name], node_curr.contact_pos_ids[name]),
                                   (node_curr.c_vel_ids[name], node_curr.q_id),
                                   (node_curr.c_vel_ids[name], node_curr.vq_id)):
                    extend_ids_lists(row_ids, col_ids, rows, cols)


class RollingWheelDynamics(WholeBodyDynamics):
    """RNEA with contact applied to the wheel, including its axle torque.

    RNEA derivatives hold external joint-local wrenches constant. The additional
    chain rule here differentiates their configuration-dependent force rotation
    and support-point lever arm. Omitting this term gives an incorrect Jacobian.
    """
    def __init__(self, wheels):
        self.wheels = wheels

    def compute_constraints(self, node_curr, node_next, w, c, model, data):
        q, v, a = q_tan2pin(w[node_curr.q_id]), w[node_curr.vq_id], w[node_curr.aq_id]
        state = self.wheels.evaluate(q, v)
        external = self.wheels.external_forces(node_curr, state, w, data)
        c[node_curr.c_dh_id] = pin.rnea(model, data, q, v, a, external)

    def compute_jacobians(self, node_curr, node_next, w, jac, model, data):
        q_rep = w[node_curr.q_id]
        q, v, a = q_tan2pin(q_rep), w[node_curr.vq_id], w[node_curr.aq_id]
        state = self.wheels.evaluate(q, v)
        external = self.wheels.external_forces(node_curr, state, w, data)
        pin.computeRNEADerivatives(model, data, q, v, a, external)
        Dq, Dv, Da = data.dtau_dq.copy(), data.dtau_dv.copy(), data.M.copy()
        for name in node_curr.contact_phase_fnames:
            if name not in self.wheels.names:
                J = pin.computeFrameJacobian(model, data, q, model.getFrameId(name), pin.LOCAL)
                jac[node_curr.c_dh_id, node_curr.forces_ids[name]] = -J[:3].T
                continue
            item = state[name]
            R, Rf, r = item["rotation"], item["force_rotation"], item["offset"]
            f_world = Rf @ w[node_curr.forces_ids[name]]
            f_local, r_local = R.T @ f_world, R.T @ r
            df_world = -pin.skew(f_world) @ item["Jproxy"][3:]
            df_local = R.T @ (df_world + pin.skew(f_world) @ item["J"][3:])
            dr_local = R.T @ (item["offset_dq"] + pin.skew(r) @ item["J"][3:])
            dwrench = np.vstack((df_local, -pin.skew(f_local) @ dr_local + pin.skew(r_local) @ df_local))
            Jlocal = pin.computeFrameJacobian(model, data, q, self.wheels.frames[name], pin.LOCAL)
            Dq -= Jlocal.T @ dwrench
            jac[node_curr.c_dh_id, node_curr.forces_ids[name]] = -item["Jmaterial"].T @ Rf
        jac[node_curr.c_dh_id, node_curr.q_id] = tangent_columns(Dq, q_rep)
        jac[node_curr.c_dh_id, node_curr.vq_id] = Dv
        jac[node_curr.c_dh_id, node_curr.aq_id] = Da
