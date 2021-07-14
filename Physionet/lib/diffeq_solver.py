import time
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from torch.distributions.multivariate_normal import MultivariateNormal


from torchdiffeq import odeint as odeint

#####################################################################################################
# for Original Latent ODE
class DiffeqSolver(nn.Module):
    def __init__(
        self,
        input_dim,
        ode_func,
        method,
        latents,
        odeint_rtol=1e-4,
        odeint_atol=1e-5,
        device=torch.device("cpu"),
    ):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
        # Decode the trajectory through ODE Solver
        """

        # first_point : 1,50,20 -> (1,batch,rec_dims)
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]
        import pdb

        pdb.set_trace()
        # pred_y : 2091,3,50,20
        pred_y = odeint(
            self.ode_func,
            first_point,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )
        pred_y = pred_y.permute(1, 2, 0, 3)  # pred_y 1,50,2,20 ->순서 바꾸는 것.

        assert torch.mean(pred_y[:, :, 0, :] - first_point) < 0.001
        assert pred_y.size()[0] == n_traj_samples
        assert pred_y.size()[1] == n_traj

        return pred_y

    def sample_traj_from_prior(
        self, starting_point_enc, time_steps_to_predict, n_traj_samples=1
    ):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(
            func,
            starting_point_enc,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y


# for anode
class DiffeqSolver_augmented(nn.Module):
    def __init__(
        self,
        input_dim,
        ode_func,
        method,
        latents,
        odeint_rtol=1e-4,
        odeint_atol=1e-5,
        device=torch.device("cpu"),
    ):
        super(DiffeqSolver_augmented, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
        # Decode the trajectory through ODE Solver
        """

        # first_point : 1,50,20 -> (1,batch,rec_dims)
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]
        # pred_y : 2091,3,50,20

        traj, batch, dims = first_point.shape
        aug = torch.zeros(traj, batch, 5).cuda()
        first_point = torch.cat([first_point, aug], 2)
        pred_y = odeint(
            self.ode_func,
            first_point,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )
        pred_y = pred_y.permute(1, 2, 0, 3)

        assert torch.mean(pred_y[:, :, 0, :] - first_point) < 0.001
        assert pred_y.size()[0] == n_traj_samples
        assert pred_y.size()[1] == n_traj

        return pred_y

    def sample_traj_from_prior(
        self, starting_point_enc, time_steps_to_predict, n_traj_samples=1
    ):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(
            func,
            starting_point_enc,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y


class DiffeqSolver_Attention(nn.Module):
    def __init__(
        self,
        input_dim,
        ode_func,
        method,
        latents,
        odeint_rtol=1e-4,
        odeint_atol=1e-5,
        device=torch.device("cpu"),
    ):
        super(DiffeqSolver_Attention, self).__init__()

        # input_dim : 2
        self.ode_method = method  # in odernn : euler #in decoder : dopri5
        self.latents = latents  # 10
        self.device = device
        self.ode_func = ode_func
        # attention init
        self.odeint_rtol = odeint_rtol  # 0.001
        self.odeint_atol = odeint_atol  # 0.0001

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
        # Decode the trajectory through ODE Solver
        """

        # first_point : 1,50,20 -> (3,batch,rec_dims)
        att0 = torch.FloatTensor([]).cuda()
        for i in range(first_point.shape[0]):
            att = np.corrcoef(np.transpose(first_point[i].cpu().detach().numpy()))
            att = np.reshape(att, (1, self.latents, self.latents))
            att = torch.Tensor(att).cuda()
            att0 = torch.cat([att0, att])
        # att0 = np.corrcoef(np.transpose(first_point[0].cpu().detach().numpy()))
        # att0 = np.reshape(att0,(1,self.latents,self.latents))
        # att0 = torch.Tensor(att0).cuda()

        # att1 = np.corrcoef(np.transpose(first_point[1].cpu().detach().numpy()))
        # att1 = np.reshape(att1,(1,self.latents,self.latents))
        # att1 = torch.Tensor(att1).cuda()

        # att2 = np.corrcoef(np.transpose(first_point[2].cpu().detach().numpy()))
        # att2 = np.reshape(att2,(1,self.latents,self.latents))
        # att2 = torch.Tensor(att2).cuda()

        # att = torch.cat([att0,att1,att2])
        # att2 = torch.Tensor(att2).cuda()

        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        self.n_size = n_traj
        xx = torch.cat([first_point, att0], dim=1)
        n_dims = first_point.size()[-1]
        # pred_y : 2091,1,70,20
        pred_y = odeint(
            self.ode_func,
            xx,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )

        pred_y_h = pred_y[:, :, :n_traj, :]
        pred_y_h = pred_y_h.permute(1, 2, 0, 3)  # pred_y 1,50,2,20 ->순서 바꾸는 것.

        assert torch.mean(pred_y_h[:, :, 0, :] - first_point) < 0.001
        assert pred_y_h.size()[0] == n_traj_samples
        assert pred_y_h.size()[1] == n_traj

        return pred_y_h

    def sample_traj_from_prior(
        self, starting_point_enc, time_steps_to_predict, n_traj_samples=1
    ):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """

        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(
            func,
            starting_point_enc,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )

        pred_y_h = pred_y[:, :, : self.n_size, :]
        pred_y_h = pred_y_h.permute(1, 2, 0, 3)
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        # pred_y = pred_y.permute(1,2,0,3)
        return pred_y_h
