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

class JointSymmetryCost(AbstractCostFunction):
    """惩罚左右关节不对称的代价函数"""
    def __init__(self, right_indices, left_indices, weight_matrix, signs):
        # 左右对应的关节索引
        self.r_idx = np.array(right_indices, dtype=int)
        self.l_idx = np.array(left_indices, dtype=int)
        # 符号标识：1 表示同向对称 (q_r = q_l), -1 表示镜像对称 (q_r = -q_l)
        self.signs = np.array(signs)
        # 惩罚权重矩阵
        self.W = weight_matrix

    def obj(self, opt_vect, node, next_node):
        # 提取当前所有 10 个驱动关节的角度
        q_joints = opt_vect[node.q_id.start + 6 : node.q_id.stop].flatten()
        
        # 计算残差: e = q_right - sign * q_left
        res = (q_joints[self.r_idx] - self.signs * q_joints[self.l_idx]).reshape(-1, 1)
        
        # 代价: Cost = 0.5 * e^T * W * e
        cost = 0.5 * res.T @ self.W @ res
        return cost[0, 0]

    def grad(self, opt_vect, cost_grad, node, next_node):
        q_joints = opt_vect[node.q_id.start + 6 : node.q_id.stop].flatten()
        res = (q_joints[self.r_idx] - self.signs * q_joints[self.l_idx]).reshape(-1, 1)
        
        # dCost/dRes = W * res
        dcost_dres = self.W @ res
        
        # 初始化关节梯度容器
        jac_joints = np.zeros(node.q_id.stop - (node.q_id.start + 6))
        
        # 根据链式法则分配梯度
        for i in range(len(self.r_idx)):
            jac_joints[self.r_idx[i]] += dcost_dres[i, 0]
            # 如果是镜像 (-1)，这里的负负得正，梯度是相同的
            jac_joints[self.l_idx[i]] -= self.signs[i] * dcost_dres[i, 0] 
            
        # 把梯度加回总梯度数组中对应的关节位置
        cost_grad[node.q_id.start + 6 : node.q_id.stop] += jac_joints