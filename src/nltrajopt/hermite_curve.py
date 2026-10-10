"""The exported manifold cubic, including its actual local acceleration."""
import numpy as np
import pinocchio as pin


def right_jacobian_derivative(x, xd):
    A, D = -pin.Motion(x).action, -pin.Motion(xd).action
    term, derivative = np.eye(6), np.zeros((6, 6))
    J, Jd = term.copy(), derivative.copy()
    for n in range(1, 81):
        derivative = (derivative@A+term@D)/(n+1)
        term = term@A/(n+1)
        J += term; Jd += derivative
        if max(np.max(abs(term)), np.max(abs(derivative))) < 1e-16:
            return J, Jd
    raise ValueError('SE3 derivative series failed to converge')


class ManifoldHermite:
    def __init__(self, model, q0, q1, v0, v1, h):
        if not np.isfinite(h) or h <= 0: raise ValueError('Positive finite duration required')
        self.model, self.q0, self.v0, self.h = model, q0, v0, h
        self.delta = pin.difference(model, q0, q1)
        # Use the same smooth right Jacobian for endpoint conversion, velocity
        # and acceleration. Small-angle dIntegrate cancellation otherwise gets
        # amplified by finite-difference collocation Jacobians.
        J, _ = right_jacobian_derivative(self.delta[:6], np.zeros(6))
        self.ve = np.array(v1, copy=True)
        self.ve[:6] = np.linalg.solve(J, v1[:6])

    def state(self, s):
        d, v0, ve, h = self.delta, self.v0, self.ve, self.h
        x = (-2*s**3+3*s**2)*d+(s**3-2*s**2+s)*h*v0+(s**3-s**2)*h*ve
        xd = ((-6*s*s+6*s)*d+(3*s*s-4*s+1)*h*v0+(3*s*s-2*s)*h*ve)/h
        xdd = ((-12*s+6)*d+(6*s-4)*h*v0+(6*s-2)*h*ve)/h**2
        q = pin.integrate(self.model, self.q0, x)
        J, Jd = right_jacobian_derivative(x[:6], xd[:6])
        v = xd.copy(); v[:6] = J@xd[:6]
        a = xdd.copy(); a[:6] = J@xdd[:6]+Jd@xd[:6]
        return q, v, a
