import torch


class AssetState:
    def __init__(self, invested_money, overall_data, window_length=50):
        self.invested_money = invested_money
        self.overall_data = overall_data
        self.window_length = window_length
        self.is_at_end_state = False

        ## number of stocks and features:
        ## Number of assets = nb_stocks + 1
        ## to account for the cash asset as well
        self.nb_stocks = self.overall_data.shape[0]
        self.nb_assets = self.nb_stocks + 1

        ## Both should be of size self.nb_stocks
        self.weight = torch.tensor([1.] + [0.] * self.nb_stocks, requires_grad=False)

        self.portfolio = invested_money
    

    def end_training_period(self, training_size):
        return int((self.overall_data.shape[1] - self.window_length) * training_size)  

    
    def _compute_asset_price_change(self, time_t):
        eps = 1e-5
        closing_prices = self.overall_data[:, time_t, 3]

        opening_prices = self.overall_data[:, time_t, 0]

        ## We add a small epsilon for smoothing to avoid division by zero
        price_change = (closing_prices + eps) / (opening_prices + eps)


        return price_change
    
    def _get_potential_portfolio_increment_after_days(self, interest, time_t):
        price_change = self._compute_asset_price_change(time_t)
        return torch.tensor([1+interest] + price_change.tolist(), requires_grad=False)
    
    ## We compute the transaction amount based on the
    ## current portfolio money and the normalized weight
    ## difference. The trade cost is also considered.
    def _transaction_amount(self, action_weights, trade_cost):
        return self.portfolio * torch.linalg.norm((action_weights - self.weight), ord=1) * trade_cost

    def _update_own_state(self, updated_weights, updated_pf):
        self.weight = updated_weights.squeeze(0)
        self.portfolio = updated_pf
    
    def get_asset_status(self):
        return self.is_at_end_state

    def update_asset_status(self):
        self.is_at_end_state = True
    
    ## We get all the assets across the specified
    ## timeframe. The timeframe is similar for all assets
    def get_data_with_time_horizon(self, t=0):
        return self.overall_data[:,t-self.window_length:t,:]

    def update_asset_with_action(self, action_weights, trade_cost, interest, time_t):

        ## Get the cost of the transaction for this update
        cost = self._transaction_amount(action_weights, trade_cost)

        ## Amount of money allocated to each asset
        updated_pf_value = self.portfolio * action_weights

        ## Amount of money allocated to each asset deducting the cost of doing this transaction
        pf_value_after_cost = updated_pf_value - torch.tensor([cost]+ [0.]*self.nb_stocks, requires_grad=False)


        pf_value_with_interest = pf_value_after_cost * self._get_potential_portfolio_increment_after_days(interest, time_t)

        total_pf_sum = torch.sum(pf_value_with_interest)

        updated_weights = pf_value_with_interest / total_pf_sum

        
        reward = (total_pf_sum - self.portfolio) #/ self.portfolio
        self._update_own_state(updated_weights, total_pf_sum)

        return reward


    def reset_state(self, weight_init, init_time):
        self.is_at_end_state = False
        init_timeframe_data = self.get_data_with_time_horizon(init_time)
        self.weight = weight_init
        self.portfolio = self.invested_money
        state = init_timeframe_data
        return state, self.is_at_end_state
        


class TradeEnvironment:

    ## We will start at time index 1 to be able
    ## to account for previous day change
    def __init__(self, asset_state, time_index, train_size=0.85,
                 trading_cost=25/1000000, interest_rate=0.25/100, window_length=50):
    
        self.asset_state = asset_state
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate
        self.window_length = window_length
        self.time_index = time_index + self.window_length
        self.end_train = self.asset_state.end_training_period(train_size)

    def reset(self, weight_init, time):
        """
        Restarts the environment with given
        initial weights and given value of portfolio
        """
        self.time_index = self.window_length + time
        return self.asset_state.reset_state(weight_init, self.time_index)
    

    def step(self, action):
        ## action is a vector of m + 1 probabilities
        reward = self.asset_state.update_asset_with_action(action, self.trading_cost,
                                                  self.interest_rate, self.time_index)
        self.time_index = self.time_index + 1
        state = self.asset_state.get_data_with_time_horizon(self.time_index)
        if self.time_index >= self.end_train:
            self.asset_state.update_asset_status()
        is_state_done = self.asset_state.get_asset_status()
        return state, reward, is_state_done
        
        