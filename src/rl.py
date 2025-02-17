from tqdm import tqdm
import torch
import numpy as np
from torch.optim import AdamW
from utils.policy import GradientPolicyPPN
from utils.dataset import Data
from utils.environment import AssetState, TradeEnvironment
from replay_buffer import PortfolioReplay
import matplotlib.pyplot as plt

NUM_ASSETS = 470
MONEY_INVESTED = 100000
TOTAL_TIME_HORIZON = 1259
TRAIN_TIME_HORIZON = 50
BATCH_SIZE = 50
N_BATCHES = 10

policy = GradientPolicyPPN(NUM_ASSETS, TRAIN_TIME_HORIZON)

def get_random_action():
    random_vec = torch.rand(NUM_ASSETS+1, requires_grad=False)
    probas = (random_vec / torch.sum(random_vec)).unsqueeze(0)
    return probas

get_random_action()

def rl_dpg(trading_data, policy, episodes, alpha=1e-4, gamma=0.99):
    start_time = 0
    asset = AssetState(MONEY_INVESTED, trading_data)

    trading_env = TradeEnvironment(asset, start_time)

    optimizer = AdamW(policy.parameters(), lr=alpha)
    stats = {'PG Loss': [], 'Returns': []}
    for episode in tqdm(range(episodes)):

        memory = PortfolioReplay(NUM_ASSETS, TOTAL_TIME_HORIZON)
        for _ in range(N_BATCHES):

            all_states = []
            all_rewards_and_weights = []
            all_rewards = []
            start_index = int(memory.draw())
            step = 0

            state, done = trading_env.reset(memory.get_W(start_index), start_index)

            while not done:
                prev_asset_weight = trading_env.asset_state.weight


                if np.random.rand() < 0.8:
                    action = policy(state.float(), prev_asset_weight.unsqueeze(1).float())
                else:
                    action = get_random_action()
                

                state, reward, done = trading_env.step(action)

                all_states.append(state)

                all_rewards.append(reward)

                all_rewards_and_weights.append((reward, prev_asset_weight))
                memory.update(start_index + step, trading_env.asset_state.weight)
                step = step + 1


            all_rewards = torch.tensor(all_rewards, requires_grad=True)
            loss = -(all_rewards).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats['PG Loss'].append(loss.item())
            stats['Returns'].append(reward)

    plt.plot(stats['PG Loss'])

data = Data().dataset
rl_dpg(data, policy, 50)
