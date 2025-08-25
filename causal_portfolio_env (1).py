import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import os
import json
import pickle
from collections import deque

def load_causal_data_for_env(data_path="data"):
    tensor_path = os.path.join(data_path, 'causal_market_tensor.npy')
    metadata_path = os.path.join(data_path, 'causal_metadata.json')
    causal_results_path = os.path.join(data_path, 'causal_results.pkl')
    if not all(os.path.exists(p) for p in [tensor_path, metadata_path, causal_results_path]):
        raise FileNotFoundError(f"Missing data in {data_path}.")
    data_tensor = np.load(tensor_path)
    with open(metadata_path, 'r') as f: metadata = json.load(f)
    with open(causal_results_path, 'rb') as f: causal_results = pickle.load(f)
    return data_tensor, metadata, causal_results

class CausalPortfolioEnv(gym.Env):
    def __init__(self, data_tensor, metadata, causal_results, lookback_window=30, causal_lookback=10, num_causal_drivers=3):
        super().__init__()
        self.data_tensor, self.metadata, self.causal_results = data_tensor, metadata, causal_results
        self.causal_lookback, self.lookback_window, self.num_causal_drivers = causal_lookback, lookback_window, num_causal_drivers
        self.T, self.N, self.F = self.data_tensor.shape
        self.tickers, self.feature_names = self.metadata['tickers'], self.metadata['feature_names']
        self.causal_drivers = self._identify_causal_drivers()
        self.transaction_cost_pct = 0.001
        self.return_history = deque(maxlen=self.lookback_window)
        self.reset()
        
        portfolio_state_size = self.N 
        causal_features_size = self.N * self.num_causal_drivers * self.causal_lookback
        lookback_features_size = self.lookback_window
        obs_dim = portfolio_state_size + causal_features_size + lookback_features_size
        
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.N,), dtype=np.float32)
        print(f"✅ Environment Ready: Focused on top {self.num_causal_drivers} drivers. Observation Space: {obs_dim} dims.")

    def _identify_causal_drivers(self):
        print("🧠 Identifying causal drivers...")
        drivers, feature_to_idx = {}, {name: i for i, name in enumerate(self.feature_names)}
        for ticker_name in self.tickers:
            asset_drivers = set()
            asset_results = self.causal_results.get(ticker_name) or self.causal_results.get(f"{ticker_name}_forced")
            if asset_results:
                for algo_result in asset_results.get('algorithms', {}).values():
                    features_used = algo_result.get('feature_names', [])
                    if 'log_return' in features_used:
                        return_idx = features_used.index('log_return')
                        
                        # --- THIS IS THE FIX ---
                        # Replaced the ambiguous 'or' with an explicit if/elif block
                        adj_matrix = None
                        if 'graph' in algo_result:
                            adj_matrix = algo_result['graph']
                        elif 'adjacency_matrix' in algo_result:
                            adj_matrix = algo_result['adjacency_matrix']
                        # --- END OF FIX ---
                        
                        if 'granger_matrix' in algo_result: 
                            adj_matrix = (algo_result.get('granger_matrix', np.array([])) < 0.05).astype(int)
                        
                        if adj_matrix is not None:
                            for i, name in enumerate(features_used):
                                if i != return_idx and adj_matrix[i, return_idx] != 0 and name in feature_to_idx: 
                                    asset_drivers.add(name)
            drivers[ticker_name] = [feature_to_idx[name] for name in asset_drivers]
            print(f"  - {ticker_name}: Found {len(drivers[ticker_name])} causal drivers.")
        return drivers

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.causal_lookback + self.lookback_window + 1
        self.weights = np.ones(self.N) / self.N
        self.prev_weights = np.ones(self.N) / self.N
        self.portfolio_value = 1.0
        self.return_history.clear()
        for _ in range(self.lookback_window): self.return_history.append(0.0)
        return self._get_observation(), {}

    def _get_observation(self):
        portfolio_state = self.weights.astype(np.float32)
        causal_history_features = []
        start, end = self.current_step - self.causal_lookback, self.current_step
        for ticker_name in self.tickers:
            driver_indices = self.causal_drivers.get(ticker_name, [])[:self.num_causal_drivers]
            asset_causal_history = self.data_tensor[start:end, self.tickers.index(ticker_name)][:, driver_indices]
            padding = np.zeros((self.causal_lookback, self.num_causal_drivers - asset_causal_history.shape[1]))
            asset_causal_history = np.hstack([asset_causal_history, padding])
            causal_history_features.append(asset_causal_history.flatten())
        lookback_vector = np.array(self.return_history, dtype=np.float32)
        observation = np.concatenate([portfolio_state, np.concatenate(causal_history_features), lookback_vector])
        return np.nan_to_num(observation.astype(np.float32))

    def step(self, action):
        self.prev_weights = self.weights.copy()
        action = np.clip(action, 0, 1)
        self.weights = action / np.sum(action) if np.sum(action) > 0 else self.prev_weights
        
        return_idx = self.feature_names.index('return')
        asset_returns = self.data_tensor[self.current_step, :, return_idx]
        portfolio_return = np.dot(self.weights, asset_returns)
        
        turnover = np.sum(np.abs(self.weights - self.prev_weights))
        transaction_costs = self.transaction_cost_pct * turnover
        net_return = portfolio_return - transaction_costs
        
        self.portfolio_value *= (1 + net_return)
        self.return_history.append(net_return)
        
        sharpe_ratio = np.mean(self.return_history) / (np.std(self.return_history) + 1e-9)
        reward = sharpe_ratio
        
        self.current_step += 1
        terminated = self.current_step >= self.T - 1
        obs = self._get_observation()
        
        return obs, reward, terminated, False, {'portfolio_value': self.portfolio_value}