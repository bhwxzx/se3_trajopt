import numpy as np
import pinocchio as pin
import nltrajopt.params as pars
from .abstract_cost import AbstractCostFunction


class ConfigurationCost(AbstractCostFunction):
    def obj(self, opt_vect, node, next_node):
        var = opt_vect[node.q_id.start + pars.SPACE_NQ : node.q_id.stop].reshape(-1, 1)
        res = self.compute_residual(var)
        cost = self.compute_cost(res)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.q_id.start + pars.SPACE_NQ : node.q_id.stop].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        cost_grad[node.q_id.start + pars.SPACE_NQ : node.q_id.stop] += jac


class JointVelocityCost(AbstractCostFunction):
    def obj(self, opt_vect, node, next_node):
        var = opt_vect[node.vq_id.start + 6 : node.vq_id.stop].reshape((-1, 1))
        res = self.compute_residual(var)
        cost = self.compute_cost(res)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.vq_id.start + 6 : node.vq_id.stop].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        cost_grad[node.vq_id.start + 6 : node.vq_id.stop] += jac


class JointAccelerationCost(AbstractCostFunction):
    def obj(self, opt_vect, node, next_node):
        var = opt_vect[node.aq_id.start + 6 : node.aq_id.stop].reshape(-1, 1)
        res = self.compute_residual(var)
        cost = self.compute_cost(res)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.aq_id.start + 6 : node.aq_id.stop].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        cost_grad[node.aq_id.start + 6 : node.aq_id.stop] += jac


class ForceCost(AbstractCostFunction):
    def obj(self, opt_vect, node, next_node):
        cost = 0
        for frame in node.contact_phase_fnames: # 只对当前接触阶段的接触力进行代价计算
            f = opt_vect[node.forces_ids[frame]]
            cost += self.compute_cost(f)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        for frame in node.contact_phase_fnames:
            f = opt_vect[node.forces_ids[frame]]
            jac = self.compute_gradient(f)
            cost_grad[node.forces_ids[frame]] = jac

class BaseRotationCost(AbstractCostFunction):
    """专门针对基座旋转（Roll, Pitch, Yaw）的代价函数"""
    def obj(self, opt_vect, node, next_node):
        # 提取基座旋转部分：从 q_id 开始后的第 3 位到第 6 位
        # [start+3 : start+6] 对应旋转向量
        var = opt_vect[node.q_id.start + 3 : node.q_id.start + 6].reshape(-1, 1)
        res = self.compute_residual(var)
        cost = self.compute_cost(res)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.q_id.start + 3 : node.q_id.start + 6].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        # 将梯度准确地填回对应的 3 个旋转维度
        cost_grad[node.q_id.start + 3 : node.q_id.start + 6] += jac

class BasePositionCost(AbstractCostFunction):
    """专门针对基座平移（X, Y, Z）的代价函数"""
    def obj(self, opt_vect, node, next_node):
        var = opt_vect[node.q_id.start : node.q_id.start + 3].reshape(-1, 1)
        res = self.compute_residual(var)
        return self.compute_cost(res)

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.q_id.start : node.q_id.start + 3].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        cost_grad[node.q_id.start : node.q_id.start + 3] += jac

class BaseRotVelocityCost(AbstractCostFunction):
    """专门针对基座角速度的代价函数"""
    def obj(self, opt_vect, node, next_node):
        var = opt_vect[node.vq_id.start + 3: node.vq_id.start + 6].reshape(-1, 1)
        res = self.compute_residual(var)
        return self.compute_cost(res)

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.vq_id.start + 3: node.vq_id.start + 6].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        cost_grad[node.vq_id.start + 3: node.vq_id.start + 6] += jac

class BasePosVelocityCost(AbstractCostFunction):
    """专门针对基座线速度的代价函数"""
    def obj(self, opt_vect, node, next_node):
        var = opt_vect[node.vq_id.start : node.vq_id.start + 3].reshape(-1, 1)
        res = self.compute_residual(var)
        return self.compute_cost(res)

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.vq_id.start : node.vq_id.start + 3].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        cost_grad[node.vq_id.start : node.vq_id.start + 3] += jac