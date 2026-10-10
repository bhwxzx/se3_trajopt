"""Numerical derivative, geometry, rolling and virtual-work checks for LW wheels."""
from pathlib import Path
import json
import sys
import unittest
import tempfile

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
# nltrajopt.params consumes command-line arguments at import time.
sys.argv = [sys.argv[0]]
from leg_to_wheel_transform import build_problem
from lw_validation_helpers import pack_saved_nodes
from nltrajopt.se3tangent import q_tan2pin
from nltrajopt.cost_models.landing import add_landing_costs


class WheelContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use retained archived nodes without running an optimizer. A small set
        # covers stance, flight, touchdown and terminal wheel support; dense
        # derivative checks do not require the complete 201-node trajectory.
        source = ROOT / "src/tests/fixtures/lw_wheel_contact_nodes.json"
        cls.saved = json.loads(source.read_text())
        full_problem, cls.robot, _ = build_problem(wheel_contact=True)
        selected = (0, 77, 78, 92, 93, 200)
        from nltrajopt.trajectory_optimization import NLTrajOpt
        cls.problem = NLTrajOpt(
            model=cls.robot.model,
            nodes=[full_problem.nodes[k] for k in selected],
            dt=full_problem.dt,
        )
        cls.problem.wheel_contact_model = full_problem.wheel_contact_model
        cls.wheels = cls.problem.wheel_contact_model
        cls.w = pack_saved_nodes(cls.problem, cls.saved)
        add_landing_costs(cls.problem, .5, 10., 5., True)

    def test_geometry_and_velocity_derivatives(self):
        rng = np.random.default_rng(72)
        for k in range(len(self.saved["solution"]["nodes"])):
            q = np.array(self.saved["solution"]["nodes"][k]["q"])
            q = pin.integrate(self.robot.model, q, rng.normal(0, .06, self.robot.model.nv))
            v = rng.normal(0, 2., self.robot.model.nv)
            values = self.wheels.evaluate(q, v, derivatives=True)
            for j in range(self.robot.model.nv):
                direction = np.eye(self.robot.model.nv)[j] * 1e-6
                plus = self.wheels.evaluate(pin.integrate(self.robot.model, q, direction), v)
                minus = self.wheels.evaluate(pin.integrate(self.robot.model, q, -direction), v)
                for name, item in values.items():
                    np.testing.assert_allclose((plus[name]["position"] - minus[name]["position"]) / 2e-6,
                                               item["Jgeometry"][:, j], atol=2e-6, rtol=2e-6)
                    np.testing.assert_allclose((plus[name]["velocity"] - minus[name]["velocity"]) / 2e-6,
                                               item["velocity_dq"][:, j], atol=4e-6, rtol=4e-6)
                    np.testing.assert_allclose(item["velocity"], item["Jmaterial"] @ v, atol=1e-12)
                    # Envelope theorem: geometric and material normal velocities agree.
                    np.testing.assert_allclose(item["Jgeometry"][2], item["Jmaterial"][2], atol=1e-11)

    def test_rolling_includes_spin_and_does_not_anchor_xy(self):
        q = pin.neutral(self.robot.model)
        v = np.zeros(self.robot.model.nv)
        v[0] = .4
        reference = self.wheels.evaluate(q, v)
        for side in ("left", "right"):
            name = side + "_wheel_contact_link"
            spin = self.robot.model.joints[self.robot.model.getJointId(side + "_wheel_joint")].idx_v
            coefficient = reference[name]["Jmaterial"][0, spin]
            self.assertGreater(abs(coefficient), .1)
            v[spin] = -.4 / coefficient
        state = self.wheels.evaluate(q, v)
        for item in state.values():
            np.testing.assert_allclose(item["velocity"], 0., atol=1e-12)
            np.testing.assert_allclose((item["Jgeometry"] @ v)[0], .4, atol=1e-12)

    def test_terminal_correction_and_saved_geometry(self):
        from lw_validation_helpers import terminal_target
        from nltrajopt.wheel_contact import WheelContactModel
        reference = np.array(self.saved["solution"]["nodes"][-1]["q"])
        restored = WheelContactModel.from_metadata(self.robot.model, self.wheels.metadata())
        for _ in range(3):
            q, report = terminal_target(self.robot.model, restored, reference)
            np.testing.assert_array_equal(q[7:], reference[7:])
            np.testing.assert_allclose(report["generalized_static_torque"][:6], 0., atol=1e-8)
            self.assertLess(abs(report["base_height_delta_m"]), .001)
            self.assertLess(abs(report["base_pitch_delta_rad"]), .002)
            for item in restored.evaluate(q).values():
                self.assertLess(abs(item["position"][2]), 1e-9)

    def test_saved_solution_keeps_moving_contact_definition(self):
        from nltrajopt.trajectory_optimization import NLTrajOpt
        problem = self.problem
        problem.sol_dict = {"model": self.robot.model.name, "solve_time": 0.,
                            "nodes": problem._decode_solution(self.w)}
        problem.info = {"status": 0}
        problem.evaluation_nums = {}
        with tempfile.TemporaryDirectory() as directory:
            path = problem.save_solution("serialization_check", directory)
            loaded = NLTrajOpt.load_solution(path)
            self.assertEqual(loaded["info"]["wheel_contact_model"], self.wheels.metadata())
            for before, after in zip(problem.sol_dict["nodes"], loaded["nodes"]):
                for name, position in before["contact_positions"].items():
                    np.testing.assert_array_equal(position, after["contact_positions"][name])

    def test_envelope_contains_loaded_collision_mesh(self):
        rng = np.random.default_rng(101)
        cdata = self.robot.collision_model.createData()
        for _ in range(20):
            q = pin.integrate(self.robot.model, pin.neutral(self.robot.model), rng.normal(0, .3, self.robot.model.nv))
            values = self.wheels.evaluate(q)
            pin.forwardKinematics(self.robot.model, self.robot.data, q)
            pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.robot.collision_model, cdata)
            for side in ("left", "right"):
                index = self.robot.collision_model.getGeometryId(side + "_wheel_link_0")
                obj = self.robot.collision_model.geometryObjects[index]
                pose = cdata.oMg[index]
                minimum = np.min(np.asarray(obj.geometry.vertices()) @ pose.rotation[2] + pose.translation[2])
                self.assertLessEqual(values[side + "_wheel_contact_link"]["position"][2], minimum + 1e-10)

    def test_full_problem_derivatives_and_virtual_work(self):
        rng = np.random.default_rng(73)
        w = self.w.copy()
        # Test nonzero base rotation and nonzero wheel spin/forces away from the nominal path.
        for node in self.problem.nodes:
            w[node.q_id] += rng.normal(0, .002, self.robot.model.nv)
            w[node.vq_id] += rng.normal(0, .1, self.robot.model.nv)
        jac = self.problem.jac_test(w)
        rows, cols = self.problem.jacobianstructure()
        sparse = np.zeros_like(jac)
        sparse[rows, cols] = self.problem.jacobian(w)
        np.testing.assert_allclose(sparse, jac, atol=1e-12)
        for _ in range(3):
            direction = rng.normal(size=len(w))
            step = 1e-6
            plus, minus = w + step * direction, w - step * direction
            numerical = (self.problem.constraints(plus) - self.problem.constraints(minus)) / (2 * step)
            np.testing.assert_allclose(numerical, jac @ direction, atol=4e-5, rtol=2e-5)
            objective_fd = (self.problem.objective(plus) - self.problem.objective(minus)) / (2 * step)
            np.testing.assert_allclose(objective_fd, self.problem.gradient(w) @ direction, atol=2e-5, rtol=1e-6)
        for k in (4, 5):
            node = self.problem.nodes[k]
            q, v, a = q_tan2pin(w[node.q_id]), w[node.vq_id], w[node.aq_id]
            values = self.wheels.evaluate(q, v)
            ext = self.wheels.external_forces(node, values, w, self.robot.data)
            rnea = pin.rnea(self.robot.model, self.robot.data, q, v, a, ext).copy()
            virtual = pin.rnea(self.robot.model, self.robot.data, q, v, a).copy()
            for name in node.contact_phase_fnames:
                virtual -= values[name]["Jmaterial"].T @ values[name]["force_rotation"] @ w[node.forces_ids[name]]
            np.testing.assert_allclose(rnea, virtual, atol=2e-11)
        # Both terminal wheels must contribute friction rows, including the second wheel.
        c = self.problem.constraints(w)
        last = self.problem.nodes[-1]
        for name in self.wheels.names:
            self.assertGreater(np.linalg.norm(c[last.c_friction_ids[name]]), 1.)


if __name__ == "__main__":
    unittest.main()
