import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from scipy.optimize import minimize
from collections import deque
import os
import json

def load_enhanced_portfolio_data(data_path="data"):
    """Load enhanced portfolio data with validation"""
    tensor_path = os.path.join(data_path, 'market_data_tensor.npy')
    metadata_path = os.path.join(data_path, 'metadata.json')

    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"Enhanced data not found in {data_path}. Please run enhanced_data_system.py first.")

    # Load tensor
    data_tensor = np.load(tensor_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return data_tensor, metadata


class EnhancedPortfolioEnv(gym.Env):
    def __init__(self, data_tensor, metadata, risk_aversion_multiplier, lookback_window=20, risk_free_rate=0.02):
      super().__init__()
      print("🏗️ Initializing Enhanced Portfolio Environment with provided data...")
      
      # --- ACCEPT DATA DIRECTLY (THE FIX) ---
      self.data_tensor = data_tensor
      self.metadata = metadata
      self.risk_aversion_multiplier = risk_aversion_multiplier # For Step 2 of our plan
      
      # --- The rest of your original setup logic remains the same ---
      self.T, self.N, self.F = self.data_tensor.shape
      self.lookback_window = min(lookback_window, 20)
      self.risk_free_rate = risk_free_rate / 252

      print(f"✅ Environment ready: {self.T} steps, {self.N} assets, {self.F} features")
      
      # Your data cleaning (assuming this method is part of your class)
      # If _aggressive_clean_data_tensor is a global function, you might need to adjust this call.
      # self.data_tensor = self._aggressive_clean_data_tensor(self.data_tensor)
      
      # Transaction costs
      self.base_transaction_cost = 0.0005
      self.impact_cost_factor = 0.0001
      
      # Risk management parameters
      self.max_weight = 0.3
      self.min_weight = 0.01
      self.max_leverage = 1.0
      self.max_drawdown_threshold = 0.15
      
      # State variables (reset in the reset method)
      self.current_step = 0
      self.weights = None
      self.portfolio_value = 1.0
      self.cash = 0.0
      self.portfolio_history = deque(maxlen=self.lookback_window)
      self.peak_value = 1.0
      
      # Observation space calculation
      market_features = min(self.N * 10, 300)
      portfolio_state = self.N + 3
      lookback_features = self.lookback_window
      obs_dim = market_features + portfolio_state + lookback_features
      
      self.observation_space = spaces.Box(
          low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
      )
      
      print(f"🎯 Observation space: {obs_dim} dimensions")
      
      # Action space
      self.action_space = spaces.Box(low=0, high=1, shape=(self.N,), dtype=np.float32)
      
      # Performance tracking
      self.performance_metrics = {
          'total_return': [], 'sharpe_ratio': [], 'max_drawdown': [], 'volatility': []
      }
    def _aggressive_clean_data_tensor(self, tensor):
        """CRITICAL FIX: Aggressive data cleaning to prevent NaN propagation"""
        print("🔧 Aggressively cleaning data tensor...")
        
        # Replace all inf values immediately
        tensor = np.where(np.isposinf(tensor), 5.0, tensor)
        tensor = np.where(np.isneginf(tensor), -5.0, tensor)
        
        # Replace all NaN values immediately
        tensor = np.where(np.isnan(tensor), 0.0, tensor)
        
        # Clip all values to conservative range
        tensor = np.clip(tensor, -5.0, 5.0)
        
        # Additional safety: replace any remaining problematic values
        T, N, F = tensor.shape
        
        for f in range(F):
            for n in range(N):
                series = tensor[:, n, f]
                
                # If series is all zeros or has extreme variance, normalize
                if np.std(series) > 3.0:
                    # Normalize extreme series
                    mean_val = np.mean(series)
                    std_val = np.std(series)
                    if std_val > 1e-8:
                        series = (series - mean_val) / std_val
                        series = np.clip(series, -3.0, 3.0)
                        tensor[:, n, f] = series
                
                # Final safety check
                tensor[:, n, f] = np.nan_to_num(tensor[:, n, f], nan=0.0, posinf=3.0, neginf=-3.0)
        
        # Ensure returns are reasonable (feature 0)
        returns = tensor[:, :, 0]
        returns = np.clip(returns, -0.2, 0.2)  # Max 20% daily return
        tensor[:, :, 0] = returns
        
        print(f"✅ Aggressive cleaning complete. NaN: {np.isnan(tensor).sum()}, Inf: {np.isinf(tensor).sum()}")
        print(f"    Value range: [{np.min(tensor):.3f}, {np.max(tensor):.3f}]")
        
        return tensor
    
    def reset(self, seed=None, options=None, start_step=None):
        """
        Resets the environment.
        
        Args:
            seed: The random seed for the environment.
            options: Additional options for resetting (part of Gymnasium API).
            start_step (int, optional): If provided, the environment will start at this specific step.
                                        Used for deterministic backtesting.
                                        If None, a random start position is chosen for training.
        """
        super().reset(seed=seed)
        
        # If a 'start_step' is given (for backtesting), use it.
        # Otherwise, pick a random start point for training.
        if start_step is not None:
            self.current_step = start_step
        else:
            # For training, start at a random point to prevent memorization.
            safe_start_range = self.T - 200 
            self.current_step = self.np_random.integers(
                low=self.lookback_window + 1, 
                high=safe_start_range
            )
            
        # Initialize portfolio state for the new episode
        self.weights = np.ones(self.N, dtype=np.float32) / self.N
        self.portfolio_value = 1.0
        self.cash = 0.0
        self.portfolio_history.clear()
        self.peak_value = 1.0
        
        # Initialize portfolio history with zeros
        for _ in range(self.lookback_window):
            self.portfolio_history.append(0.0)
        
        # Get the initial observation
        obs = self._get_observation()
        obs = self._validate_observation(obs)
        
        return obs, {}
    
    def _get_observation(self):
        """Simplified observation to prevent NaN propagation"""
        # Bounds checking
        step_idx = min(max(self.current_step, 0), self.T - 1)
        
        # Get current market data with aggressive safety
        current_data = self.data_tensor[step_idx, :, :].copy()
        current_data = np.nan_to_num(current_data, nan=0.0, posinf=3.0, neginf=-3.0)
        current_data = np.clip(current_data, -3.0, 3.0)
        
        # Simplified market features - just take first 10 features per asset
        market_features_per_asset = min(10, self.F - 1)
        market_obs = current_data[:, 1:market_features_per_asset+1].flatten()  # Skip returns
        
        # Pad or trim to consistent size
        target_market_size = min(self.N * 10, 300)
        if len(market_obs) > target_market_size:
            market_obs = market_obs[:target_market_size]
        elif len(market_obs) < target_market_size:
            padding = np.zeros(target_market_size - len(market_obs))
            market_obs = np.concatenate([market_obs, padding])

        log_portfolio_value = np.log(np.maximum(self.portfolio_value, 1e-9))
        # Portfolio state - simplified
        portfolio_state = np.concatenate([
            self.weights,
            [log_portfolio_value, self.cash, self._calculate_simple_drawdown()]
        ])
        
        # Lookback features - just portfolio returns
        lookback_returns = list(self.portfolio_history)
        if len(lookback_returns) != self.lookback_window:
            # Ensure exact size
            if len(lookback_returns) > self.lookback_window:
                lookback_returns = lookback_returns[-self.lookback_window:]
            else:
                padding_needed = self.lookback_window - len(lookback_returns)
                lookback_returns = [0.0] * padding_needed + lookback_returns
        
        lookback_features = np.array(lookback_returns, dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([
            market_obs.astype(np.float32), 
            portfolio_state.astype(np.float32), 
            lookback_features
        ])
        
        return observation

    def _calculate_simple_drawdown(self):
        """Simplified drawdown calculation"""
        if self.peak_value <= 0:
            return 0.0
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        return np.clip(drawdown, 0.0, 1.0)

    def _validate_observation(self, obs):
        """CRITICAL FIX: Aggressive observation validation"""
        # Replace any NaN or inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Clip to observation space bounds
        obs = np.clip(obs, self.observation_space.low[0], self.observation_space.high[0])
        
        # Ensure correct shape
        expected_shape = self.observation_space.shape[0]
        if obs.shape[0] != expected_shape:
            if obs.shape[0] > expected_shape:
                obs = obs[:expected_shape]
            else:
                padding = np.zeros(expected_shape - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, padding])
        
        # Final type check
        obs = obs.astype(np.float32)
        
        # Double check for any remaining issues
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"⚠️ WARNING: Still found invalid values in observation!")
            obs = np.zeros_like(obs)  # Emergency fallback
        
        return obs

    def _apply_risk_constraints(self, raw_action):
        """Simplified risk constraint application"""
        # Clean action input
        action = np.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=0.0)
        action = np.clip(action, 0.0, 1.0)
        
        # Normalize weights
        weight_sum = np.sum(action)
        if weight_sum > 1e-8:
            weights = action / weight_sum
        else:
            weights = np.ones(self.N, dtype=np.float32) / self.N
        
        # Apply weight constraints
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / np.sum(weights)  # Renormalize
        
        return weights

    def _calculate_transaction_costs(self, new_weights):
        """Simplified transaction cost calculation"""
        if self.weights is None:
            return 0.0, 0.0
        
        weight_changes = np.abs(new_weights - self.weights)
        turnover = np.sum(weight_changes)
        
        # Simple linear cost
        total_costs = turnover * self.base_transaction_cost
        return np.clip(total_costs, 0.0, 0.01), np.clip(turnover, 0.0, 2.0)

    def step(self, action):
        # Your action validation and risk constraints code at the top remains the same
        target_weights = self._apply_risk_constraints(action)
        
        # Calculate costs and returns
        transaction_costs, turnover = self._calculate_transaction_costs(target_weights)
        asset_returns = self.data_tensor[self.current_step, :, 0].copy()
        portfolio_return = np.dot(target_weights, asset_returns)
        net_return = portfolio_return - transaction_costs
        
        # --- THIS IS THE FINAL, CORRECTED REWARD LOGIC ---
        # The primary reward is the real, per-step profit.
        # We add a tiny, targeted penalty to discourage hyperactive trading.
        turnover_penalty_multiplier = 0.01
        final_reward = net_return - (turnover_penalty_multiplier * turnover)
        # --- END OF REWARD LOGIC ---

        # Update portfolio state
        self.portfolio_value *= (1 + net_return)
        self.weights = target_weights.copy()
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        # Prepare for next step
        self.current_step += 1
        terminated = bool(self.current_step >= self.T - 1)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'turnover': turnover,
        }
        
        obs = self._get_observation()
        obs = self._validate_observation(obs)
        
        return obs, final_reward, terminated, False, info

    def get_performance_summary(self):
        """Get simplified performance summary"""
        if not self.performance_metrics['total_return']:
            return {}
        
        final_return = self.performance_metrics['total_return'][-1]
        
        return {
            'total_return_pct': final_return * 100,
            'final_portfolio_value': self.portfolio_value,
            'final_weights': self.weights.copy() if self.weights is not None else np.ones(self.N) / self.N,
            'total_steps': self.current_step
        }