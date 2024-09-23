# from replay_buffer import ReplayBuffer
# from network import Q_network
# from lightning import LightningModule
# from src.utils.policy import GradientPolicy

import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from utils.policy import GradientPolicyPPN
from utils.dataset import StockDataset
from utils.environment import AssetState, TradeEnvironment


NUM_ASSETS = 16
MONEY_INVESTED = 1e9


         
def reinforce(trading_data, policy, episodes, alpha=1e-4, gamma=0.99):
    start_time = 0
    asset = AssetState(MONEY_INVESTED, trading_data)
    trading_env = TradeEnvironment(asset, start_time)
    optim = AdamW(policy.parameters(), lr=alpha)
    stats = {'PG Loss': [], 'Returns': []}


    for episode in tqdm(range(episodes)):
        init_state = trading_env.reset()
        transitions = []
        done = False

        while not done:
            action = policy(init_state)








dataset = StockDataset().get_assets_at_random(NUM_ASSETS)
num_assets, time_horizon, attr = dataset.shape
policy = GradientPolicyPPN(num_assets, time_horizon)