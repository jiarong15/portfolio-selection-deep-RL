import numpy as np

class PortfolioReplay:
    def __init__(self, num_assets, total_time_horizon, window_length=50):
        w_init = np.array(np.array([1] + [0] * num_assets))
        self.memory = np.transpose(np.array([w_init]*total_time_horizon))  
        # self.sample_bias = sample_bias
        self.total_steps = total_time_horizon * 0.6
        self.batch_size = window_length

    def get_W(self, time_t):
        #return the weight from the PVM at time t 
        return self.memory[:, time_t]

    def update(self, time_t, weight):
        #update the weight at time t
        self.memory[:, time_t] = weight

    def draw(self, beta=5e-4):
        '''
        returns a valid step so you can 
        get a training batch starting at this step
        '''
        while True:
            z_var = np.random.geometric(p=beta)
            train_batch = self.total_steps - self.batch_size + 1 - z_var
            if 0 <= train_batch:
                return train_batch

    # def __init__(self, capacity):
    #     self.buffer = deque(maxlen=capacity)
    
    # def __len__(self):
    #     return len(self.buffer)
    
    # def append(self, experience):
    #     self.buffer.append(experience)

    # def sample(self, batch_size):
    #     return random.sample(self.buffer, batch_size)
    

