import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

import lib.utils as utils

#####################################################################################################


class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)


#####################################################################################################


class ODEFunc_w_Poisson(ODEFunc):
    def __init__(
        self,
        input_dim,
        latent_dim,
        ode_func_net,
        lambda_net,
        device=torch.device("cpu"),
    ):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc_w_Poisson, self).__init__(
            input_dim, latent_dim, ode_func_net, device
        )

        self.latent_ode = ODEFunc(
            input_dim=input_dim,
            latent_dim=latent_dim,
            ode_func_net=ode_func_net,
            device=device,
        )

        self.latent_dim = latent_dim
        self.lambda_net = lambda_net
        # The computation of poisson likelihood can become numerically unstable.
        # The integral lambda(t) dt can take large values. In fact, it is equal to the expected number of events on the interval [0,T]
        # Exponent of lambda can also take large values
        # So we divide lambda by the constant and then multiply the integral of lambda by the constant
        self.const_for_lambda = torch.Tensor([100.0]).to(device)

    def extract_poisson_rate(self, augmented, final_result=True):
        y, log_lambdas, int_lambda = None, None, None

        assert augmented.size(-1) == self.latent_dim + self.input_dim
        latent_lam_dim = self.latent_dim // 2

        if len(augmented.size()) == 3:
            int_lambda = augmented[:, :, -self.input_dim :]
            y_latent_lam = augmented[:, :, : -self.input_dim]

            log_lambdas = self.lambda_net(y_latent_lam[:, :, -latent_lam_dim:])
            y = y_latent_lam[:, :, :-latent_lam_dim]

        elif len(augmented.size()) == 4:
            int_lambda = augmented[:, :, :, -self.input_dim :]
            y_latent_lam = augmented[:, :, :, : -self.input_dim]

            log_lambdas = self.lambda_net(y_latent_lam[:, :, :, -latent_lam_dim:])
            y = y_latent_lam[:, :, :, :-latent_lam_dim]

        # Multiply the intergral over lambda by a constant
        # only when we have finished the integral computation (i.e. this is not a call in get_ode_gradient_nn)
        if final_result:
            int_lambda = int_lambda * self.const_for_lambda

        # Latents for performing reconstruction (y) have the same size as latent poisson rate (log_lambdas)
        assert y.size(-1) == latent_lam_dim

        return y, log_lambdas, int_lambda, y_latent_lam

    def get_ode_gradient_nn(self, t_local, augmented):
        y, log_lam, int_lambda, y_latent_lam = self.extract_poisson_rate(
            augmented, final_result=False
        )
        dydt_dldt = self.latent_ode(t_local, y_latent_lam)

        log_lam = log_lam - torch.log(self.const_for_lambda)
        return torch.cat((dydt_dldt, torch.exp(log_lam)), -1)


class ODEFunc_Attention(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        ode_func_net,
        n_traj_samples,
        device=torch.device("cpu"),
    ):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc_Attention, self).__init__()

        self.input_dim = input_dim
        self.device = device
        self.n_traj_samples = n_traj_samples

        utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net
        self.gradient_net_at = ode_func_net
        self.fc = torch.nn.Linear(50, 20)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, t_local, y, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """

        # grad = self.get_ode_gradient_nn(t_local, y)

        grad, grad2 = self.get_ode_gradient_nn(t_local, y)
        # 4,50,20 + 4,20,20 -> out shape : 2,50,20
        # import pdb;pdb.set_trace()
        if backwards:
            grad = -grad
            grad2 = -grad2
        size0 = int(y.shape[1])
        size = int(y.shape[2])

        # a_y = y[:,size0-size:]
        # a_s = self.softmax(a_y)
        # print(grad2.shape)
        size = int(grad2.shape[1])
        size2 = int(grad2.shape[2])
        self.fc = torch.nn.Linear(
            size * size2, size2 * size2
        ).cuda()  # 50*20== 1000 -> 20*20 ==400
        # grad2 : 4,20,50 -> 4,1000
        
        grad2_fc = torch.reshape(grad2, (self.n_traj_samples, size * size2)).cuda()

        # import pdb; pdb.set_trace()
        # grad2_fc.cuda()

        grad2 = self.fc(grad2_fc)
        # 4,1000 -> 4,400
        grad2 = torch.reshape(grad2, (self.n_traj_samples, size2, size2))

        # import pdb; pdb.set_trace()
        out = torch.cat([grad, grad2], dim=1)
        return out

    def get_ode_gradient_nn(self, t_local, y):
        # import pdb; pdb.set_trace()
        size0 = int(y.shape[1])
        size = int(y.shape[2])
        h_y = y[:, : size0 - size, :]
        a_y = y[:, size0 - size :]
        a_s = self.softmax(a_y)
        h_yT = torch.transpose(h_y, 1, 2)
        h_ = torch.matmul(a_y, h_yT)
        h_ = torch.transpose(h_, 1, 2)
        return self.gradient_net(h_), self.gradient_net_at(h_)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)
