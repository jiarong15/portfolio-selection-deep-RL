import numpy as np


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
        self.weight = self._initial_portfolio_weights()
        self.portfolio = invested_money
    

    def end_training_period(self, training_size):
        return int((self.overall_data.shape[1] - self.window_length) * training_size)  

    ## With index 0 as the cash asset, our starting
    ## weight distribution is such that all our assets
    ## are just cash before investing into any other stocks
    ## The distribution is that we have 100% cash
    def _initial_portfolio_weights(self):
        return np.array([1.] + [0.] * self.nb_stocks)
    
    def _compute_asset_price_change(self, time_t):
        closing_prices = self.overall_data[:, time_t, 3]
        opening_prices = self.overall_data[:, time_t, 0]
        price_change = closing_prices / opening_prices

        return price_change
    
    def _get_potential_portfolio_increment_after_days(self, interest, time_t):
        price_change = self._compute_asset_price_change(time_t)
        return np.array([1+interest] + price_change.tolist())
    
    def _transaction_amount(self, action_weights, trade_cost):
        return self.portfolio * np.linalg.norm((action_weights - self.weight), ord=1) * trade_cost

    def _update_own_state(self, updated_weights, updated_pf):
        self.weight = updated_weights
        self.portfolio = updated_pf
    
    def get_asset_status(self):
        return self.is_at_end_state

    def update_asset_status(self):
        self.is_at_end_state = True
    
    ## We get all the assets across the specified
    ## timeframe. The timeframe is similar for all assets
    def get_data_with_time_horizon(self, t=0):
        return self.overall_data[:,t-self.window_length:t,:]
    
    ## Return total amount of money earned from
    ## investing proportions of money by the RL trading
    ## strategy
    def get_portfolio_value(self):
        return self.portfolio


    def update_asset_with_action(self, action_weights, trade_cost, interest, time_t):
        cost = self._transaction_amount(action_weights, trade_cost)
        updated_pf_value = self.portfolio * action_weights
        pf_value_after_cost = updated_pf_value - np.array([cost]+ [0.]*self.nb_stocks)
        pf_value_with_interest = pf_value_after_cost * self._get_potential_portfolio_increment_after_days(interest, time_t)
        total_pf_sum = np.sum(pf_value_with_interest)

        updated_weights = pf_value_with_interest / total_pf_sum
        reward = (total_pf_sum - self.portfolio) / self.portfolio

        self._update_own_state(updated_weights, total_pf_sum)

        return reward


    def reset_state(self, init_time):
        self.is_at_end_state = False
        init_timeframe_data = self.get_data_with_time_horizon(init_time)
        self.weight = self._initial_portfolio_weights()
        self.portfolio = self.invested_money
        state = init_timeframe_data
        return state, self.is_at_end_state
        


class TradeEnvironment:

    ## We will start at time index 1 to be able
    ## to account for previous day change
    def __init__(self, asset_state, time_index, train_size=0.85,
                 trading_cost=0.25/100, interest_rate=0.05/100, window_length=50):
    
        self.asset_state = asset_state
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate
        self.window_length = window_length
        self.time_index = time_index + self.window_length
        self.end_train = self.asset_state.end_training_period(train_size)

    def reset(self):
        """
        Restarts the environment with given
        initial weights and given value of portfolio
        """
        self.time_index = self.window_length
        return self.asset_state.reset_state(self.time_index)
    

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
        
        
        
        
        

        
        
 