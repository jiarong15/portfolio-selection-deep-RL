from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from utils.policy import GradientPolicyPPN
from utils.dataset import StockDataset
from torch.utils.data import DataLoader
from utils.environment import AssetState, TradeEnvironment
from replay_buffer import PortfolioReplay

NUM_ASSETS = 16
MONEY_INVESTED = 1e7
TIME_HORIZON = 1259
BATCH_SIZE = 50

policy = GradientPolicyPPN(NUM_ASSETS, TIME_HORIZON)

def get_random_action():
    random_vec = np.random.rand(NUM_ASSETS + 1)
    return random_vec / np.sum(random_vec)

def rl_dpg(trading_data, policy, episodes, alpha=1e-4, gamma=0.99):
    start_time = 0
    asset = AssetState(MONEY_INVESTED, trading_data)
    train_dataloader = DataLoader(StockDataset(), batch_size=BATCH_SIZE, shuffle=True)
    trading_env = TradeEnvironment(asset, start_time)

    optim = AdamW(policy.parameters(), lr=alpha)
    stats = {'PG Loss': [], 'Returns': []}

    for episode in tqdm(range(episodes)):
        memory = PortfolioReplay(NUM_ASSETS, TIME_HORIZON)
        state = trading_env.reset()
        all_states = []
        all_rewards = []


        for batch in train_dataloader:

            start_index = memory.draw()
            done = False
            while not done:

                if np.random.rand() < 0.8:
                    action = policy(state)
                else:
                    action = get_random_action()
                
                state, reward, done = trading_env.step(action)
                all_states.append(state)
                all_rewards.append(reward)
                



                # transitions.append((state, action, reward))
            #     memory.insert([state, action, reward])

            # if memory.can_sample(batch_size):
            #     state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)
            #     qsa_b = q_network(state_b).gather(1, action_b)
            #     next_action_b = policy(next_state_b, epsilon)
            #     next_qsa_b = target_q_network(next_state_b).gather(1, next_action_b)
            #     target_b = reward_b + ~done_b * gamma * next_qsa_b
            #     loss = F.mse_loss(qsa_b, target_b)
            #     q_network.zero_grad()
            #     loss.backward()
            #     optim.step()

            #     stats['MSE Loss'].append(loss.item())


# rl_dpg(np.random.rand(16, 1259), policy, 5)
np.load('stock_data/all_stocks_5yr.csv')