import torch
from typing import Union


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, radius=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.rho = rho
        self.radius = radius
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.list = []

        self.step_size = 1
        self.gamma = 0.5

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        grad_norm = self._grad_norm()

        #
        # def project(self, param_name, param_data):
        #     r = param_data - self.param_backup[param_name]  # ew
        #     norm = torch.norm(r)
        #     if norm > self.radius:
        #         r = self.radius * r / norm
        #     return self.param_backup[param_name] + r
        # assert torch.abs(grad_norm)>1e-12

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm+ 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]= p.data.clone()

                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)

                # e_w.clamp_(-self.radius, self.radius)
                norm = torch.norm(e_w)
                if norm > self.radius:
                    e_w = self.radius * e_w / norm

                p.add_(e_w)  # climb to the local maximum "w + e(w)"
            # import ipdb
            # ipdb.set_trace()

        if zero_grad: self.zero_grad()



    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is None:
                    continue
                if torch.is_tensor(self.state[p]) == False:
                    continue

                p.data = self.state[p]  # get back to "w" from "w + e(w)"

        #g = self.param_groups[0]['params'][-2].grad

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        # gs = self.param_groups[0]['params'][-2].grad
        # import numpy as np
        # import ipdb
        # ipdb.set_trace()
        #thea = np.dot(g, gs.T()) / (torch.norm(g) * torch.norm(gs))
        #gv = gs - torch.norm(gs) * thea * g / torch.norm(g)


        # with open('/root/grover-main/step3.txt', 'a', encoding='utf-8') as f:
        #    f.write("%.4f"%gv)
        #    f.write(',')

    @torch.no_grad()
    def step(self, i, epoch, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        if epoch % self.step_size == 0 and i == 0:
            self.step_rho(epoch)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):

        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism

        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step_rho(self, epoch):
        self.rho = self.rho * pow(self.gamma, epoch / self.step_size)
        self.radius = self.radius * pow(self.gamma, epoch / self.step_size)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



class GraphSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, arg, adaptive=True, **kwargs):
        assert arg.rho >= 0.0, f"Invalid rho, should be non-negative: {arg.rho}"

        defaults = dict(rho=arg.rho, adaptive=adaptive, **kwargs)
        super(GraphSAM, self).__init__(params, defaults)
        self.rho = arg.rho
        self.radius = arg.radius
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.list = []

        self.gradh = None
        self.grads = None
        self.alpha = arg.alpha
        self.device = torch.device(f'cuda:{arg.gpu}')
        self.step_size = arg.epoch_steps
        self.gamma = arg.gamma


    @torch.no_grad()
    def first_step(self, zero_grad=False, i=0):

        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        gh_par = self.param_groups[0]["params"]

        # 保存 每个epoch 第一个step的梯度
        if i == 0:
           self.gh_grad = []
           for k in range(len(gh_par)):
                p = gh_par[k]
                if p.grad is None:
                    self.gh_grad.append(p.grad)
                else:
                    self.gh_grad.append(p.grad.cpu())

            # import ipdb
            # ipdb.set_trace()
        # g_h2 = alpha * g_h1 + (1-alpha) * g_s1
        else:
            for k in range(len(gh_par)):
                if self.gh_grad[k] is not None:
                    self.gh_grad[k] = (1-self.alpha)*(self.gs_grad[k]/(torch.norm(self.gs_grad[k]+ 1e-8))) + self.alpha*self.gh_grad[k]
                    #self.gh_grad[k] = self.gh_grad[k]/(1-pow(self.alpha,i))
        for j in range (len(gh_par)):
            grad = self.gh_grad[j]
            p = gh_par[j]
            if grad is None: continue
            grad = grad.to(self.device)

            self.state[p] = p.data.clone()

            e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * grad * scale.to(p)
            # e_w.clamp_(-self.radius, self.radius)
            norm = torch.norm(e_w)
            if norm > self.radius:
                e_w = self.radius * e_w / norm
            # e_w = self.radius * e_w / norm
            p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()



    @torch.no_grad()
    def second_step(self, zero_grad=False):

        gh_par = self.param_groups[0]["params"]
        self.gs_grad = []
        for k in range(len(gh_par)):
            p = gh_par[k]
            if p.grad is None:
                self.gs_grad.append(p.grad)
            else:
                self.gs_grad.append(p.grad.cpu())

            if p.grad is None:
                continue
            if torch.is_tensor(self.state[p]) == False:
                continue

            p.data = self.state[p]  # get back to "w" from "w + e(w)"

        #g = self.param_groups[0]['params'][-2].grad

        self.base_optimizer.step()  # do the actual "sharpness-aware" update


        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, i, epoch, closure=None, loss=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        if i == 0:
            loss.backward()
        #if epoch % self.step_size == 0 and self.rho > 0.001 and i==0:
        if epoch % self.step_size == 0 and i == 0:
            self.step_rho(epoch)
            # if self.rho < 0.001:
            #     self.rho = 0.001
            #     self.radius = 0.001
            print(self.rho)
        self.first_step(zero_grad=True, i=i)
        self.loss = closure()
        self.second_step()


    def step_rho(self, epoch):
        self.rho = self.rho * pow(self.gamma, epoch/self.step_size)
        self.radius =self.radius * pow(self.gamma, epoch/self.step_size)


    def get_loss(self):
        return self.loss

    def _grad_norm(self):

        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism

        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class LookSAM(torch.optim.Optimizer):

    def __init__(self, k, alpha, model, base_optimizer, criterion, rho=0.05, **kwargs):

        """
        LookSAM algorithm: https://arxiv.org/pdf/2203.02714.pdf
        Optimization algorithm that capable of simultaneously minimizing loss and loss sharpness to narrow
        the generalization gap.
        :param k: frequency of SAM's gradient calculation (default: 10)
        :param model: your network
        :param criterion: your loss function
        :param base_optimizer: optimizer module (SGD, Adam, etc...)
        :param alpha: scaling factor for the adaptive ratio (default: 0.7)
        :param rho: radius of the l_p ball (default: 0.1)
        :return: None
        Usage:
            model = YourModel()
            criterion = YourCriterion()
            base_optimizer = YourBaseOptimizer
            ...
            for train_index, data in enumerate(loader):
                loss = criterion(model(samples), targets)
                loss.backward()
                optimizer.step(t=train_index, samples=samples, targets=targets, zero_grad=True)
            ...
        """

        defaults = dict(alpha=alpha, rho=rho, **kwargs)
        self.model = model
        super(LookSAM, self).__init__(self.model.parameters(), defaults)

        self.k = k
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.criterion = criterion

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.criterion = criterion

    @staticmethod
    def normalized(g):
        return g / g.norm(p=2)

    def step(self, t, closure=None, zero_grad=False):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        for group in self.param_groups:
            scale = group['rho'] / (self._grad_norm() + 1e-12) if not t % self.k else None

            if not t % self.k:
                for index_p, p in enumerate(group['params']):
                    if p.grad is None:
                        continue

                    self.state[p] = p.data.clone()
                    self.state[f'old_grad_p_{index_p}'] = p.grad.clone()

                    with torch.no_grad():
                        e_w = p.grad * scale.to(p)
                        p.add_(e_w)

            if not t % self.k:
                closure()

            for index_p, p in enumerate(group['params']):
                if not t % self.k:
                    old_grad_p = self.state[f'old_grad_p_{index_p}']
                    g_grad_norm = LookSAM.normalized(old_grad_p)
                    g_s_grad_norm = LookSAM.normalized(p.grad)
                    self.state[f'gv_{index_p}']['gv'] = torch.sub(p.grad, p.grad.norm(p=2) * torch.sum(
                        g_grad_norm * g_s_grad_norm) * g_grad_norm)

                else:
                    with torch.no_grad():
                        gv = self.state[f'gv_{index_p}']['gv']
                        p.grad.add_(self.alpha.to(p) * (p.grad.norm(p=2) / gv.norm(p=2)) * gv)

                p.data = self.state[p]

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )

        return norm




