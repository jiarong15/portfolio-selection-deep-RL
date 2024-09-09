from replay_buffer import ReplayBuffer
from network import Q_network
from lightning import LightningModule
from src.utils.policy import GradientPolicy
import torch

def polyak_average(network, target_network, tau=0.01):
    for qp, tp in zip(network.parameters(), target_network.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)


class DDPG(LightningModule):
    def __init__(self, gamma=0.99, batch_size=512, samples_per_epoch=64,
                 loss_fn=torch.nn.HuberLoss, optim=torch.optim.adam):
        super().__init__()
        self.q_net = Q_network(batch_size)
        self.policy = GradientPolicy()


         