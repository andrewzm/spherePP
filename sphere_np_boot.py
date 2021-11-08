import sys
import numpy as np
import torch
import json
from torch.autograd import Variable
import math
from random import choices
import random
  

def exponential_map_sd(x, v):
    r"""Exponential map on \mathbb{S}^D.
    Args:
        x: points on \mathbb{S}^D, embedded in \mathbb{R}^{D+1}
        v: vectors in the tangent space T_x \mathbb{S}^D
    Returns:
        Image of exponential map
    """
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    return x * np.cos(v_norm) + v / v_norm * np.sin(v_norm)


def sample_sd(d=2, num_samples=1):
    xs = np.random.randn(d + 1, num_samples)
    xs /= np.linalg.norm(xs, axis=0)
    return xs.transpose()


def vMF_dens(x, mu, kappa):
    if kappa > 0:
        pdf_constant = kappa / ((2 * np.pi) * (1. - np.exp(-2. * kappa)))
        return pdf_constant * np.exp(kappa * (mu.dot(x) - 1.))
    else:
        return 1. / (4 * np.pi)


def sample_vMF(mu, kappa, num_samples):
    """Generate num_samples N-dimensional samples from von Mises Fisher
    distribution around center mu \in R^N with concentration kappa.
    """
    dim = len(mu)
    result = np.zeros((num_samples, dim))
    for nn in range(num_samples):
        # sample offset from center (on sphere) with spread kappa
        w = _sample_weight(kappa, dim)

        # sample a point v on the unit sphere that's orthogonal to mu
        v = _sample_orthonormal_to(mu)

        # compute new point
        result[nn, :] = v * np.sqrt(1. - w ** 2) + w * mu

    return result


def _sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)
    x = (1. - b) / (1. + b)
    c = kappa * x + dim * np.log(1 - x ** 2)

    while True:
        z = np.random.beta(dim / 2., dim / 2.)
        w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
            return w


def _sample_orthonormal_to(mu):
    """Sample point on sphere orthogonal to mu."""
    v = np.random.randn(mu.shape[0])
    proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
    orthto = v - proj_mu_v
    return orthto / np.linalg.norm(orthto)


def spherical_to_euclidean(sph_coords):
    if sph_coords.ndim == 1:
        sph_coords = np.expand_dims(sph_coords, 0)
    theta, phi = np.split(sph_coords, 2, 1)
    return np.concatenate((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ), 1)


def euclidean_to_spherical(euc_coords):
    if euc_coords.ndim == 1:
        euc_coords = np.expand_dims(euc_coords, 0)
    x, y, z = np.split(euc_coords, 3, 1)
    return np.concatenate((
        math.atan2(y, x),
        math.atan2(np.sqrt(x ** 2 + y ** 2) / z)
    ), 1)


def batch_outer(x, y):
    return torch.bmm(x.unsqueeze(2), y.unsqueeze(1))


def softplus(x):
    return torch.logaddexp(x, torch.tensor([0.]))


def softplus_inv(x):
    return torch.log(-1. + torch.exp(x))


def softmax(x):
    ex = torch.exp(x - torch.max(x))
    return ex / torch.sum(ex)


class RadialFlowVmfMixBase:
    def __init__(self, N, inputs, mean_dir_list, conc_list, mix_prob_list):
        self._N = N  # no. of layers
        self._inputs = inputs
        self._mean_dir_list = mean_dir_list
        self._conc_list = conc_list
        self._mix_prob_list = mix_prob_list
        self.RF = RadialFlow(N, inputs)
        self._ncomp = len(mix_prob_list)
        self._log_pdf_const_list = [self.vmf_log_pdf_const(conc) for conc in conc_list]
        self._pdf_const_list = [self.vmf_pdf_const(conc) for conc in conc_list]

    def vmf_pdf_const(self, conc):
        num = conc
        den = 2 * np.pi * (np.exp(conc) - np.exp(-conc))
        return num / den

    def vmf_log_pdf_const(self, conc):
        pdf_const = self.vmf_pdf_const(conc)
        return np.log(pdf_const)

    def vmf_density(self, x, mean_dir, conc, pdf_const):
        return pdf_const * np.exp(conc * np.dot(x, mean_dir))

    def vmf_log_density(self, x, mean_dir, conc, log_pdf_const):
        return conc * np.dot(x, mean_dir) + log_pdf_const

    def loss(self, theta_all, mu_all, beta_all, prop_mini_batch=.2):
        ljs_inputs = 0.
        for x in self._inputs:
            u = torch.rand(1)
            if u < prop_mini_batch:
                z, ljs = self.RF.serial(theta_all, mu_all, beta_all, x)
                # print(ljs)

                dens_base = 0.
                for k in range(self._ncomp):
                    dens_base += self._mix_prob_list[k] * self.vmf_density(x, self._mean_dir_list[k],
                                                                           self._conc_list[k],
                                                                           self._pdf_const_list[k])

                log_dens_base = torch.log(torch.tensor(dens_base))
                # print(log_dens_base)

                ljs_inputs += ljs + log_dens_base

        return - ljs_inputs

    def log_density(self, theta_all, mu_all, beta_all, x):
        ljs = 0.

        for theta, mu, beta in zip(theta_all, mu_all, beta_all):
            x, lj = self.RF.comp_flow(theta, mu, beta, x)
            ljs += lj

        dens_base = 0
        x = x.detach().numpy()
        for k in range(self._ncomp):
            dens_base += self._mix_prob_list[k] * self.vmf_density(x, self._mean_dir_list[k], self._conc_list[k],
                                                                   self._pdf_const_list[k])
        log_dens_base = torch.log(torch.tensor(dens_base))

        ljs += log_dens_base

        return ljs


class RadialFlow:
    def __init__(self, N, inputs):
        self._N = N  # no. of layers
        self._inputs = inputs

    def comp_flow(self, theta, mu, beta, x):
        """WARNING WORKS ONLY FOR p = 1"""
        x = x.to(torch.float32)
        nx = len(x)
        theta = softmax(theta)
        mu = mu / torch.linalg.norm(mu)
        beta = torch.exp(beta)

        p = len(theta)
        if p > 1:
            sys.exit("This code only works for p = 1")
        alpha = torch.arccos(torch.matmul(x, mu.T))

        num_e = mu - torch.cos(alpha) * x
        den_e = 1. / torch.sin(alpha)
        e = num_e * den_e

        f_prime = - torch.sin(alpha) * torch.exp(beta * (torch.cos(alpha) - 1))

        v_theta = - theta * f_prime * e

        alpha_theta = (torch.linalg.norm(v_theta, dim=1)).unsqueeze(0).T

        e_theta = v_theta / alpha_theta

        H1 = batch_outer(e_theta, e_theta)
        H2 = alpha_theta * torch.cos(alpha_theta) / torch.sin(alpha_theta)
        IdMatrices = torch.eye(3).reshape((1, 3, 3)).repeat(nx, 1, 1)
        H3 = IdMatrices - batch_outer(x, x) - batch_outer(e_theta, e_theta)
        H_theta = H1 + H2.reshape(nx, 1, 1) * H3

        K_sum = torch.zeros(nx, 3, 3)
        for i in range(p):
            f1 = - torch.sin(alpha[:, i]) * torch.exp(beta[i] * (torch.cos(alpha[:, i]) - 1))
            f2_1 = - torch.cos(alpha[:, i]) * torch.exp(beta[i] * (torch.cos(alpha[:, i]) - 1))
            f2_2 = beta[i] * torch.sin(alpha[:, i]) ** 2 * torch.exp(beta[i] * (torch.cos(alpha[:, i]) - 1))
            f2 = torch.add(f2_1, f2_2)

            K1 = f2.reshape(nx, 1, 1) * batch_outer(e, e)
            K2 = f1 * torch.cos(alpha[:, i]) / torch.sin(alpha[:, i])
            K3 = IdMatrices - batch_outer(x, x) - batch_outer(e, e)
            K = K1 + K2.reshape(nx, 1, 1) * K3
            K_sum += theta[i] * K

        # transformed
        v_theta_norm = (torch.linalg.norm(v_theta, dim=1)).unsqueeze(0).T
        z = x * torch.cos(v_theta_norm) + v_theta / v_theta_norm * torch.sin(v_theta_norm)

        # log jacobian
        lj1 = 2 * torch.log(torch.sin(v_theta_norm) / v_theta_norm)
        lj2 = torch.logdet(batch_outer(x, x) + H_theta + K_sum).reshape(nx, 1)
        lj = lj1 + lj2

        return z, lj

    def serial(self, theta_all, mu_all, beta_all, x):
        ljs = 0.
        for theta, mu, beta in zip(theta_all, mu_all, beta_all):
            x, lj = self.comp_flow(theta, mu, beta, x)
            ljs += lj
        return x, ljs

    def loss(self, theta_all, mu_all, beta_all, prop_mini_batch=.3):
        # print(beta_all)
        # print(mu_all)
        x_all = self._inputs
        nsamp = np.random.binomial(len(x_all), prop_mini_batch)
        idx = np.random.choice(len(x_all),
                               size=nsamp,
                               replace=False)
        x = x_all[idx, :]
        z, ljs = self.serial(theta_all, mu_all, beta_all, x)

        return - sum(ljs)

    def weighted_loss(self, theta_all, mu_all, beta_all, weights, prop_mini_batch=.3):
        ljs_inputs = 0.

        for x, w in zip(self._inputs, weights):
            u = torch.rand(1)
            if u < prop_mini_batch:
                z, ljs = self.serial(theta_all, mu_all, beta_all, x)
                ljs_inputs += ljs * w
                # z1 = z.detach().numpy()
        return - ljs_inputs

    def log_density(self, theta_all, mu_all, beta_all, x):
        ljs = 0.
        for theta, mu, beta in zip(theta_all, mu_all, beta_all):
            x, lj = self.comp_flow(theta, mu, beta, x)
            ljs += lj
        return ljs


rows = []
with open('pacific.csv') as f:
    for line in f:
        # strip whitespace
        line = line.strip()
        # separate the columns
        line = line.split(',')
        # save the line for use later
        rows.append(line)

data = []
for i in range(2, len(rows)):
    if rows[i][0] != rows[i - 1][0]:
        data.append(rows[i - 1])


theta_vec1 = []
theta_vec2 = []
for d in data:
    theta_vec1.append(float(d[6][:-1]))
    theta_vec2.append(float(d[7][:-1]))
theta_vec2 = [360. - theta_vec2[i] for i in range(len(theta_vec2))]

theta_vec1 = [(theta_vec1[i] + 90.) * np.pi / 180. for i in range(len(theta_vec1))]
theta_vec2 = [(theta_vec2[i] + 180.) * np.pi / 180. for i in range(len(theta_vec2))]
theta_vec = [np.array([theta1, theta2]) for (theta1, theta2) in zip(theta_vec1, theta_vec2)]

print("### Bootstrapping ###")

for i in range(50):

    print("Bootstrap sample ", i+1)
    
    np.random.seed(20000+i)
    torch.manual_seed(20000+i)
    random.seed(20000+i)
   
    
    nobs = len(theta_vec)
    n_samp = np.random.poisson(nobs, size=1)[0]

    theta_vec_samp = choices(theta_vec, k=n_samp)

    #theta_vec_list = [theta.tolist() for theta in theta_vec]

    inputs = np.array([spherical_to_euclidean(theta_vec_samp[i])[0] for i in range(len(theta_vec_samp))])

    inputs = torch.from_numpy(inputs)

    N = 30
    p = 1
    RF = RadialFlow(N, inputs)

    mu_all = Variable(0.5 * torch.randn((N, p, 3)), requires_grad=True)
    theta_all = Variable(0.5 * torch.randn((N, p)), requires_grad=True)
    beta_all = Variable(0.5 * torch.randn((N, p)), requires_grad=True)

    optimizer = torch.optim.SGD([theta_all, mu_all, beta_all], lr=1e-5)
    loss_list = [10 ** 10]

    for t in range(100):
        # Forward pass:

        loss = RF.loss(theta_all, mu_all, beta_all)
        # print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dens_list = []

    ngrids = 200

    theta_vec1 = np.linspace(.001, np.pi - .001, ngrids)
    phi_vec1 = np.linspace(.001, 2 * np.pi - .001, ngrids)

    X_sph = np.array([(x, y) for x in theta_vec1 for y in phi_vec1])
    X = [spherical_to_euclidean(x_sph)[0] for x_sph in X_sph]
    X = np.array(X)
    X = torch.from_numpy(X)

    dens = torch.exp(RF.log_density(theta_all, mu_all, beta_all, X))

    dens = torch.mul(dens, n_samp)

    dens_list = np.concatenate((X_sph, dens.detach()), axis=1).tolist()

    write_path = 'pacific_end_dens_est_radial_np_v' + str(i+20000) + '.json'
    with open(write_path, 'w') as f:
        json.dump(dens_list, f)
