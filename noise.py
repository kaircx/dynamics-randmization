import torch

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.6, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = torch.zeros_like(self.mu) if self.x0 is None else self.x0

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.randn_like(self.mu)
        self.x_prev = x
        return x

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
