"""Smooth finite-width LW wheel support geometry on a horizontal plane.

The support shape encloses a cylinder (radius r, half width h), with its
axis along the wheel link's local Y. A tiny radial regularizer and an axial
ellipsoid regularize the cylinder's flat-tread contact selection. The model
is axisymmetric: geometric contact is independent of spin, while material
velocity and virtual work include the actuated wheel spin.

Forces keep the legacy contact-frame ORIENTATION for saved-file compatibility;
their application point moves to the computed support point on the wheel.
"""

import numpy as np
import pinocchio as pin


class WheelContactModel:
    def __init__(self, model, radius=0.10950001, half_width=0.02500001,
                 axial_offset=0.0045, axial_smoothing=0.01, radial_smoothing=1e-6):
        self.model = model
        self.data = model.createData()
        self.radius = radius
        self.half_width = half_width
        self.axial_offset = axial_offset
        self.axial_smoothing = axial_smoothing
        self.radial_smoothing = radial_smoothing
        self.names = tuple(f"{side}_wheel_contact_link" for side in ("left", "right"))
        self.frames = {name: model.getFrameId(name.replace("_contact", "")) for name in self.names}
        for name, fid in self.frames.items():
            if fid >= model.nframes or not model.existFrame(name):
                raise ValueError(f"Missing wheel/contact frame: {name}")

    def metadata(self):
        return {"kind": "smooth_finite_width_wheel_flat_ground_v1", "radius_m": self.radius,
                "half_width_m": self.half_width, "left_axial_offset_m": self.axial_offset,
                "right_axial_offset_m": -self.axial_offset,
                "axial_smoothing": self.axial_smoothing, "radial_smoothing": self.radial_smoothing,
                "normal": [0, 0, 1], "wheel_axis_in_link": [0, 1, 0],
                "force_components": "legacy contact-frame orientation; application at moving wheel support point",
                "velocity_constraint": "world material velocity = center velocity + omega cross support offset = 0",
                "geometry_note": "Smooth conservative cylinder envelope; nominal additional height is 0.25 mm."}

    @classmethod
    def from_metadata(cls, model, metadata):
        if metadata["kind"] != "smooth_finite_width_wheel_flat_ground_v1":
            raise ValueError("Unknown wheel contact model version")
        return cls(model, metadata["radius_m"], metadata["half_width_m"],
                   metadata["left_axial_offset_m"], metadata["axial_smoothing"], metadata["radial_smoothing"])

    def evaluate(self, q, v=None, derivatives=False):
        model, data = self.model, self.data
        v = np.zeros(model.nv) if v is None else np.asarray(v)
        if derivatives:
            pin.computeForwardKinematicsDerivatives(model, data, q, v, np.zeros(model.nv))
        else:
            pin.computeJointJacobians(model, data, q)
            pin.forwardKinematics(model, data, q, v)
        pin.updateFramePlacements(model, data)
        result = {}
        n = np.array([0., 0., 1.])
        for name, fid in self.frames.items():
            pose = data.oMf[fid]
            R = pose.rotation.copy()
            axis = R[:, 1]
            beta = axis[2]
            radial2 = self.radial_smoothing ** 2
            axial2 = self.axial_smoothing ** 2
            s = np.sqrt(max(0., 1 - beta * beta) + radial2)
            b = np.sqrt(beta * beta + axial2)
            u = (1 + radial2) * n - beta * axis
            z = beta * axis + axial2 * n
            d = self.axial_offset if name.startswith("left") else -self.axial_offset
            offset = d * axis - self.radius * u / s - self.half_width * z / b
            # d(offset)/d(axis), for variations of the unit axle direction.
            B = np.outer(axis, n) + beta * np.eye(3)
            D = (d * np.eye(3) + self.radius * B / s
                 - self.radius * np.outer(u, beta * n) / s ** 3
                 - self.half_width * B / b
                 + self.half_width * np.outer(z, beta * n) / b ** 3)
            J = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED).copy()
            dr = D @ (-pin.skew(axis) @ J[3:])
            Jmaterial = J[:3] - pin.skew(offset) @ J[3:]
            Jgeometry = J[:3] + dr
            motion = pin.getFrameVelocity(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
            velocity = motion.linear + np.cross(motion.angular, offset)
            proxy_id = model.getFrameId(name)
            force_rotation = data.oMf[proxy_id].rotation.copy()
            Jproxy = pin.getFrameJacobian(model, data, proxy_id, pin.LOCAL_WORLD_ALIGNED).copy()
            values = {"position": pose.translation.copy() + offset, "offset": offset,
                      "rotation": R, "force_rotation": force_rotation, "axis": axis,
                      "J": J, "Jproxy": Jproxy, "Jmaterial": Jmaterial,
                      "Jgeometry": Jgeometry, "offset_dq": dr, "velocity": velocity}
            if derivatives:
                local = pin.getFrameVelocity(model, data, fid, pin.LOCAL)
                dq, dv = pin.getFrameVelocityDerivatives(model, data, fid, pin.LOCAL)
                dlinear = R @ dq[:3] - R @ pin.skew(local.linear) @ dv[3:]
                dangular = R @ dq[3:] - R @ pin.skew(local.angular) @ dv[3:]
                values["velocity_dq"] = dlinear - pin.skew(offset) @ dangular + pin.skew(motion.angular) @ dr
            result[name] = values
        return result

    def external_forces(self, node, state, w, data):
        """External joint-local wrenches, including moving application points."""
        model = self.model
        external = [pin.Force.Zero() for _ in range(model.njoints)]
        for name in node.contact_phase_fnames:
            force = w[node.forces_ids[name]]
            if name in self.names:
                item = state[name]
                fid = self.frames[name]
                local_f = item["rotation"].T @ item["force_rotation"] @ force
                local_r = item["rotation"].T @ item["offset"]
                wrench = pin.Force(local_f, np.cross(local_r, local_f))
            else:
                fid = model.getFrameId(name)
                wrench = pin.Force(force, np.zeros(3))
            frame = model.frames[fid]
            external[frame.parentJoint] += frame.placement.act(wrench)
        return external
