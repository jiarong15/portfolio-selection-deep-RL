import numpy as np
import torch

class PortfolioReplay:
    def __init__(self, num_assets, total_time_horizon, window_length=50):
        w_init = torch.tensor([1.] + [0.] * num_assets, requires_grad=False)
        self.memory = self.create_tensor_copies(num_assets + 1, total_time_horizon, w_init)
        self.total_steps = total_time_horizon * 0.6
        self.batch_size = window_length

    def create_tensor_copies(self, length_of_sub_tensor, horizon_length, original_tensor):
        final_tensor = torch.empty(horizon_length, length_of_sub_tensor, requires_grad=False)
        for i in range(horizon_length):
            destination_tensor = torch.empty(length_of_sub_tensor, requires_grad=False)
            destination_tensor.copy_(original_tensor)
            final_tensor[i] = destination_tensor
        final_tensor = final_tensor.transpose(1,0)
        return final_tensor


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