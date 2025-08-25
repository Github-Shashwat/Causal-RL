import yfinance as yf
import numpy as np
import pandas as pd
import ta
import os
from datetime import datetime, timedelta
import warnings
import requests
import time
import json
warnings.filterwarnings('ignore')

class CausalPortfolioDataCollector:
    def __init__(self, tickers, start_date='2018-01-01', end_date='2024-01-01'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = {}
        self.macro_data = {}
        self.market_data = {}
        self.processed_data = None
        self.feature_names = []
        
        # Define macroeconomic and market indicators
        self.macro_indicators = {
            # Market Indices
            '^GSPC': 'sp500',        # S&P 500
            '^IXIC': 'nasdaq',       # NASDAQ
            '^DJI': 'dow',           # Dow Jones
            '^RUT': 'russell2000',   # Russell 2000
            
            # Volatility & Risk
            '^VIX': 'vix',           # VIX Volatility Index
            '^MOVE': 'move',         # MOVE Bond Volatility (if available)
            
            # Commodities
            'GC=F': 'gold',          # Gold Futures
            'SI=F': 'silver',        # Silver Futures
            'CL=F': 'oil',           # Crude Oil Futures
            'NG=F': 'natural_gas',   # Natural Gas Futures
            
            # Currencies
            'DX-Y.NYB': 'usd_index', # USD Index
            'EURUSD=X': 'eurusd',    # EUR/USD
            'USDJPY=X': 'usdjpy',    # USD/JPY
            
            # Interest Rates & Bonds
            '^TNX': 'treasury_10y',  # 10-Year Treasury Yield
            '^FVX': 'treasury_5y',   # 5-Year Treasury Yield
            '^IRX': 'treasury_3m',   # 3-Month Treasury Bill
            'TLT': 'long_bond_etf',  # Long-term Bond ETF
            'HYG': 'high_yield',     # High Yield Corporate Bond ETF
            
            # Sector ETFs for sector rotation signals
            'XLF': 'financials',     # Financial Select Sector
            'XLK': 'technology',     # Technology Select Sector
            'XLE': 'energy',         # Energy Select Sector
            'XLV': 'healthcare',     # Healthcare Select Sector
            'XLI': 'industrials',    # Industrials Select Sector
            'XLU': 'utilities',      # Utilities Select Sector
            'XLRE': 'real_estate',   # Real Estate Select Sector
            'XLY': 'consumer_disc',  # Consumer Discretionary
            'XLP': 'consumer_staples', # Consumer Staples
            'XLB': 'materials',      # Materials Select Sector
            
            # Economic Indicators (ETFs/proxies)
            'IEF': 'intermediate_bonds', # Intermediate Treasury ETF
            'TIP': 'tips',           # TIPS (Inflation-Protected Securities)
            'EMB': 'emerging_bonds', # Emerging Market Bonds
            'EEM': 'emerging_markets', # Emerging Markets Equity
            'VNQ': 'reits',          # REITs
            
            # Credit & Risk Indicators
            'LQD': 'investment_grade', # Investment Grade Corporate Bonds
            'JNK': 'junk_bonds',     # High Yield Bonds
        }
    
    def _normalize_timezone(self, df):
        """Normalize DataFrame index to remove timezone information"""
        if df.index.tz is not None:
            # Convert to timezone-naive
            df.index = df.index.tz_convert(None)
        return df
    
    def check_network_connectivity(self):
        """Check if we can connect to Yahoo Finance"""
        try:
            response = requests.get('https://finance.yahoo.com', timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def download_stock_data(self, max_retries=3, delay=2):
        """Download stock data for portfolio tickers"""
        print(f"📊 Downloading stock data for {len(self.tickers)} tickers...")
        
        for i, ticker in enumerate(self.tickers):
            success = False
            
            for attempt in range(max_retries):
                try:
                    print(f"  [{i+1}/{len(self.tickers)}] Downloading {ticker}... (attempt {attempt+1})")
                    
                    stock = yf.Ticker(ticker)
                    df = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)
                    
                    if df.empty:
                        print(f"  ❌ No data found for {ticker}")
                        break
                    
                    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df.columns for col in expected_columns):
                        print(f"  ❌ Missing required columns for {ticker}")
                        break
                    
                    # Normalize timezone
                    df = self._normalize_timezone(df)
                    df = df.dropna()
                    
                    if len(df) < 100:
                        print(f"  ❌ Insufficient data for {ticker} ({len(df)} rows)")
                        break
                        
                    self.raw_data[ticker] = df
                    print(f"  ✅ {ticker}: {len(df)} rows")
                    success = True
                    break
                    
                except Exception as e:
                    print(f"  ⚠️  Attempt {attempt+1} failed for {ticker}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    continue
            
            if not success:
                print(f"  ❌ All attempts failed for {ticker}")
            time.sleep(0.5)
        
        print(f"✅ Successfully downloaded stock data for {len(self.raw_data)} tickers")
    
    def download_macro_data(self, max_retries=3, delay=2):
        """Download macroeconomic and market indicator data"""
        print(f"📈 Downloading macro/market data for {len(self.macro_indicators)} indicators...")
        
        for i, (symbol, name) in enumerate(self.macro_indicators.items()):
            success = False
            
            for attempt in range(max_retries):
                try:
                    print(f"  [{i+1}/{len(self.macro_indicators)}] Downloading {name} ({symbol})... (attempt {attempt+1})")
                    
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)
                    
                    if df.empty:
                        print(f"  ❌ No data found for {name}")
                        break
                    
                    # For macro indicators, we mainly need Close prices
                    if 'Close' not in df.columns:
                        print(f"  ❌ No Close price for {name}")
                        break
                    
                    # Normalize timezone
                    df = self._normalize_timezone(df)
                    df = df.dropna()
                    
                    if len(df) < 50:  # More lenient for macro data
                        print(f"  ❌ Insufficient data for {name} ({len(df)} rows)")
                        break
                        
                    self.macro_data[name] = df
                    print(f"  ✅ {name}: {len(df)} rows")
                    success = True
                    break
                    
                except Exception as e:
                    print(f"  ⚠️  Attempt {attempt+1} failed for {name}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    continue
            
            if not success:
                print(f"  ❌ All attempts failed for {name}")
            time.sleep(0.3)  # Shorter delay for macro data
        
        print(f"✅ Successfully downloaded macro data for {len(self.macro_data)} indicators")
    
    def generate_synthetic_macro_data(self):
        """Generate synthetic macro data for testing"""
        print("🔧 Generating synthetic macro data...")
        
        np.random.seed(42)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        dates = dates[dates.dayofweek < 5]  # Remove weekends
        # Ensure timezone-naive
        dates = dates.tz_localize(None) if dates.tz is not None else dates
        
        for name, _ in self.macro_indicators.items():
            print(f"  Generating synthetic data for {name}...")
            
            n_days = len(dates)
            
            if 'vix' in name.lower():
                # VIX-like data (mean-reverting, higher volatility)
                base_level = 20
                returns = np.random.normal(-0.001, 0.05, n_days)  # Mean reverting
                prices = base_level * np.exp(returns.cumsum())
                prices = np.maximum(prices, 10)  # Floor at 10
            elif 'treasury' in name.lower() or 'yield' in name.lower():
                # Interest rate data (bounded, mean-reverting)
                base_rate = 2.5
                returns = np.random.normal(0, 0.002, n_days)
                prices = base_rate + returns.cumsum()
                prices = np.maximum(prices, 0.1)  # Floor at 0.1%
            elif 'gold' in name.lower() or 'silver' in name.lower():
                # Commodity prices (higher volatility)
                base_price = 1800 if 'gold' in name.lower() else 25
                returns = np.random.normal(0.0002, 0.015, n_days)
                prices = base_price * np.exp(returns.cumsum())
            elif 'oil' in name.lower():
                # Oil prices (very volatile, mean-reverting)
                base_price = 70
                returns = np.random.normal(0, 0.03, n_days)
                prices = base_price * np.exp(returns.cumsum())
                prices = np.maximum(prices, 20)  # Floor at $20
            else:
                # General market indicators
                base_price = np.random.uniform(100, 4000)
                returns = np.random.normal(0.0005, 0.012, n_days)
                prices = base_price * np.exp(returns.cumsum())
            
            # Create OHLC data
            volume = np.random.uniform(1000000, 50000000, n_days)
            high = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
            low = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
            open_price = np.roll(prices, 1)
            open_price[0] = prices[0]
            
            df = pd.DataFrame({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': prices,
                'Volume': volume
            }, index=dates[:n_days])
            
            self.macro_data[name] = df
            print(f"  ✅ {name}: {len(df)} synthetic rows")
        
        print(f"✅ Generated synthetic macro data for {len(self.macro_data)} indicators")
    
    def calculate_macro_features(self):
        """Calculate features from macroeconomic data"""
        print("🔧 Calculating macro features...")
        
        macro_features = {}
        
        for name, df in self.macro_data.items():
            try:
                feature_df = pd.DataFrame(index=df.index)
                
                # Price-based features
                feature_df[f'{name}_close'] = df['Close']
                feature_df[f'{name}_return'] = df['Close'].pct_change()
                feature_df[f'{name}_log_return'] = np.log(df['Close'] / df['Close'].shift(1))
                
                # Volatility
                feature_df[f'{name}_vol_5d'] = feature_df[f'{name}_return'].rolling(5).std()
                feature_df[f'{name}_vol_20d'] = feature_df[f'{name}_return'].rolling(20).std()
                
                # Moving averages and momentum
                feature_df[f'{name}_sma_10'] = df['Close'].rolling(10).mean()
                feature_df[f'{name}_sma_20'] = df['Close'].rolling(20).mean()
                feature_df[f'{name}_momentum_5d'] = df['Close'].pct_change(5)
                feature_df[f'{name}_momentum_20d'] = df['Close'].pct_change(20)
                
                # Relative strength
                feature_df[f'{name}_rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
                
                # Z-score (standardized levels)
                rolling_mean = df['Close'].rolling(60).mean()
                rolling_std = df['Close'].rolling(60).std()
                feature_df[f'{name}_zscore'] = (df['Close'] - rolling_mean) / rolling_std
                
                # Regime indicators
                feature_df[f'{name}_above_sma20'] = (df['Close'] > feature_df[f'{name}_sma_20']).astype(int)
                
                # Volume features (if available)
                if 'Volume' in df.columns:
                    feature_df[f'{name}_volume_change'] = df['Volume'].pct_change()
                    vol_ma = df['Volume'].rolling(20).mean()
                    feature_df[f'{name}_volume_ratio'] = df['Volume'] / vol_ma
                
                # Clean data
                feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
                feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
                
                macro_features[name] = feature_df
                print(f"  ✅ {name}: {feature_df.shape[1]} features calculated")
                
            except Exception as e:
                print(f"  ❌ Error calculating features for {name}: {str(e)}")
                continue
        
        self.macro_features = macro_features
        print(f"✅ Calculated macro features for {len(macro_features)} indicators")
        return macro_features
    
    def calculate_cross_asset_features(self):
        """Calculate cross-asset and lagged features for causal relationships"""
        print("🔧 Calculating cross-asset causal features...")
        
        if not self.raw_data:
            print("❌ No stock data available for cross-asset features")
            return {}
        
        cross_features = {}
        
        # Create a combined dataframe of all asset returns
        all_returns = pd.DataFrame()
        for ticker, df in self.raw_data.items():
            # Ensure timezone normalization
            df = self._normalize_timezone(df)
            all_returns[f'{ticker}_return'] = df['Close'].pct_change()
            all_returns[f'{ticker}_vol'] = df['Close'].pct_change().rolling(20).std()
        
        # Add macro returns - ensure timezone consistency
        for name, df in self.macro_data.items():
            df = self._normalize_timezone(df)
            all_returns[f'{name}_return'] = df['Close'].pct_change()
        
        # Calculate cross-correlations and lagged features
        for ticker in self.raw_data.keys():
            try:
                # Ensure timezone normalization for stock data
                stock_df = self._normalize_timezone(self.raw_data[ticker].copy())
                feature_df = pd.DataFrame(index=stock_df.index)
                
                # Lagged returns of other assets
                other_tickers = [t for t in self.raw_data.keys() if t != ticker]
                
                for other_ticker in other_tickers:
                    if other_ticker in self.raw_data:
                        other_df = self._normalize_timezone(self.raw_data[other_ticker].copy())
                        other_returns = other_df['Close'].pct_change()
                        
                        # Align indices before operations
                        common_index = feature_df.index.intersection(other_returns.index)
                        if len(common_index) > 0:
                            # Lagged returns (1-day and 5-day lags)
                            other_returns_aligned = other_returns.reindex(feature_df.index)
                            feature_df[f'{other_ticker}_lag1'] = other_returns_aligned.shift(1)
                            feature_df[f'{other_ticker}_lag5'] = other_returns_aligned.shift(5)
                            
                            # Rolling correlation
                            asset_returns = stock_df['Close'].pct_change()
                            feature_df[f'corr_{other_ticker}_30d'] = asset_returns.rolling(30).corr(other_returns_aligned)
                
                # Lagged macro features
                for macro_name in self.macro_data.keys():
                    if macro_name in self.macro_data:
                        macro_df = self._normalize_timezone(self.macro_data[macro_name].copy())
                        macro_returns = macro_df['Close'].pct_change()
                        macro_returns_aligned = macro_returns.reindex(feature_df.index)
                        feature_df[f'{macro_name}_lag1'] = macro_returns_aligned.shift(1)
                        feature_df[f'{macro_name}_lag5'] = macro_returns_aligned.shift(5)
                
                # Market beta (rolling)
                if 'sp500' in self.macro_data:
                    market_df = self._normalize_timezone(self.macro_data['sp500'].copy())
                    market_returns = market_df['Close'].pct_change()
                    market_returns_aligned = market_returns.reindex(feature_df.index)
                    asset_returns = stock_df['Close'].pct_change()
                    
                    # Calculate rolling beta
                    rolling_cov = asset_returns.rolling(60).cov(market_returns_aligned)
                    rolling_var = market_returns_aligned.rolling(60).var()
                    feature_df[f'beta_sp500'] = rolling_cov / rolling_var
                
                # VIX-related features
                if 'vix' in self.macro_data:
                    vix_df = self._normalize_timezone(self.macro_data['vix'].copy())
                    vix_level = vix_df['Close'].reindex(feature_df.index)
                    vix_change = vix_level.pct_change()
                    
                    feature_df['vix_level'] = vix_level
                    feature_df['vix_change'] = vix_change
                    feature_df['vix_regime'] = (vix_level > vix_level.rolling(60).quantile(0.75)).astype(int)
                
                # Sector rotation signals (if sector ETFs available)
                sector_etfs = ['financials', 'technology', 'energy', 'healthcare']
                available_sectors = [s for s in sector_etfs if s in self.macro_data]
                
                if len(available_sectors) >= 2:
                    # Calculate sector momentum
                    for sector in available_sectors:
                        sector_df = self._normalize_timezone(self.macro_data[sector].copy())
                        sector_momentum = sector_df['Close'].pct_change(20)
                        sector_momentum_aligned = sector_momentum.reindex(feature_df.index)
                        feature_df[f'{sector}_momentum'] = sector_momentum_aligned
                
                # Clean data
                feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
                feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
                feature_df = feature_df.dropna()
                
                cross_features[ticker] = feature_df
                print(f"  ✅ {ticker}: {feature_df.shape[1]} cross-asset features")
                
            except Exception as e:
                print(f"  ❌ Error calculating cross-asset features for {ticker}: {str(e)}")
                continue
        
        self.cross_features = cross_features
        print(f"✅ Calculated cross-asset features for {len(cross_features)} tickers")
        return cross_features
    
    def calculate_enhanced_stock_features(self):
        """Enhanced stock feature calculation with regime awareness"""
        print("🔧 Calculating enhanced stock features...")
        
        processed_data = {}
        
        for ticker, df in self.raw_data.items():
            try:
                # Ensure timezone normalization
                stock_df = self._normalize_timezone(df.copy())
                print(f"  🔍 {ticker}: Starting with {len(stock_df)} raw rows")
                
                # Basic price and volume features
                stock_df['return'] = stock_df['Close'].pct_change()
                stock_df['log_return'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))
                stock_df['volume_change'] = stock_df['Volume'].pct_change()
                
                volume_ma = stock_df['Volume'].rolling(10, min_periods=1).mean()  # Reduced from 20
                stock_df['volume_ratio'] = (stock_df['Volume'] / volume_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)
                
                # Technical indicators with reduced periods
                stock_df['rsi'] = ta.momentum.RSIIndicator(stock_df['Close'], window=10).rsi().fillna(50)  # Reduced from 14
                
                try:
                    macd = ta.trend.MACD(stock_df['Close'], window_fast=8, window_slow=16, window_sign=6)  # Reduced periods
                    stock_df['macd'] = macd.macd().fillna(0)
                    stock_df['macd_signal'] = macd.macd_signal().fillna(0)
                    stock_df['macd_histogram'] = macd.macd_diff().fillna(0)
                except:
                    stock_df['macd'] = 0.0
                    stock_df['macd_signal'] = 0.0
                    stock_df['macd_histogram'] = 0.0
                
                # Moving averages with reduced windows
                stock_df['sma_10'] = stock_df['Close'].rolling(10, min_periods=1).mean()  # Reduced from 20
                stock_df['sma_20'] = stock_df['Close'].rolling(20, min_periods=5).mean()  # Reduced from 50
                stock_df['ema_8'] = stock_df['Close'].ewm(span=8, min_periods=1).mean()   # Reduced from 12
                
                # Bollinger Bands with reduced period
                try:
                    bb = ta.volatility.BollingerBands(stock_df['Close'], window=15, window_dev=2)  # Reduced from 20
                    stock_df['bb_high'] = bb.bollinger_hband()
                    stock_df['bb_low'] = bb.bollinger_lband()
                    stock_df['bb_mid'] = bb.bollinger_mavg()
                    
                    bb_range = stock_df['bb_high'] - stock_df['bb_low']
                    stock_df['bb_width'] = (bb_range / stock_df['bb_mid']).replace([np.inf, -np.inf], np.nan)
                    stock_df['bb_position'] = ((stock_df['Close'] - stock_df['bb_low']) / bb_range).replace([np.inf, -np.inf], np.nan)
                except:
                    stock_df['bb_high'] = stock_df['Close'] * 1.02
                    stock_df['bb_low'] = stock_df['Close'] * 0.98
                    stock_df['bb_mid'] = stock_df['Close']
                    stock_df['bb_width'] = 0.04
                    stock_df['bb_position'] = 0.5
                
                # Volatility and risk measures with reduced windows
                stock_df['volatility'] = stock_df['return'].rolling(10, min_periods=3).std()      # Reduced from 20
                stock_df['volatility_long'] = stock_df['return'].rolling(30, min_periods=10).std() # Reduced from 60
                
                # Momentum with reduced periods
                stock_df['momentum_3'] = stock_df['Close'].pct_change(3)   # Reduced from 5
                stock_df['momentum_5'] = stock_df['Close'].pct_change(5)   # Reduced from 10
                stock_df['momentum_10'] = stock_df['Close'].pct_change(10) # Reduced from 20
                
                # Price ratios
                stock_df['high_low_ratio'] = (stock_df['High'] / stock_df['Low']).replace([np.inf, -np.inf], np.nan)
                stock_df['close_to_high'] = (stock_df['Close'] / stock_df['High']).replace([np.inf, -np.inf], np.nan)
                stock_df['close_to_low'] = (stock_df['Close'] / stock_df['Low']).replace([np.inf, -np.inf], np.nan)
                
                # ATR with reduced period
                high_low = stock_df['High'] - stock_df['Low']
                high_close = abs(stock_df['High'] - stock_df['Close'].shift())
                low_close = abs(stock_df['Low'] - stock_df['Close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                stock_df['atr'] = true_range.rolling(10, min_periods=3).mean()  # Reduced from 14
                
                # Regime features with reduced windows
                returns_mean = stock_df['return'].rolling(10, min_periods=3).mean()
                returns_std = stock_df['return'].rolling(10, min_periods=3).std()
                stock_df['sharpe_10'] = (returns_mean / returns_std).replace([np.inf, -np.inf], np.nan)  # Reduced from 20
                
                rolling_max = stock_df['Close'].rolling(120, min_periods=30).max()  # Reduced from 252
                stock_df['drawdown'] = (stock_df['Close'] / rolling_max - 1).fillna(0)
                
                # ADX with reduced period
                try:
                    adx = ta.trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=10)  # Reduced from 14
                    stock_df['adx'] = adx.adx().fillna(20)
                except:
                    stock_df['adx'] = 20.0
                
                print(f"  🔍 {ticker}: Before cleaning: {len(stock_df)} rows")
                
                # More aggressive cleaning and filling
                stock_df = stock_df.replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN values more aggressively
                numeric_cols = stock_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if stock_df[col].isnull().all():
                        stock_df[col] = 0  # Fill completely null columns with 0
                    else:
                        # Fill with forward/backward fill first
                        stock_df[col] = stock_df[col].fillna(method='ffill').fillna(method='bfill')
                        # Fill any remaining with column mean
                        if stock_df[col].isnull().any():
                            stock_df[col] = stock_df[col].fillna(stock_df[col].mean())
                        # Fill any remaining (if mean is NaN) with 0
                        if stock_df[col].isnull().any():
                            stock_df[col] = stock_df[col].fillna(0)
                
                # Drop only the first few rows that might have NaN due to rolling calculations
                stock_df = stock_df.iloc[5:]  # Skip first 5 rows instead of using dropna()
                
                print(f"  🔍 {ticker}: After cleaning: {len(stock_df)} rows")
                
                if len(stock_df) >= 20:  # Reduced from 50
                    processed_data[ticker] = stock_df
                    print(f"  ✅ {ticker}: {len(stock_df)} rows with enhanced features")
                else:
                    print(f"  ❌ Insufficient data for {ticker} ({len(stock_df)} rows)")
                    
            except Exception as e:
                print(f"  ❌ Error processing {ticker}: {str(e)}")
                import traceback
                print(f"  🔍 Traceback: {traceback.format_exc()}")
                continue
        
        self.processed_data = processed_data
        print(f"✅ Enhanced stock features calculated for {len(processed_data)} tickers")
        return processed_data


# --- REPLACE your old merge_all_features method with this ---
# --- REPLACE your old merge_all_features method with this FINAL version ---

    def merge_all_features(self):
        """Merge stock features with macro and cross-asset features with robust cleaning."""
        print("🔄 Merging all feature sets...")
        
        if not self.processed_data:
            print("❌ No processed stock data available to merge.")
            return {}
        
        merged_data = {}
        
        for ticker, stock_df in self.processed_data.items():
            try:
                # Start with the processed stock data
                merged_df = stock_df.copy()

                # Add macro features
                if hasattr(self, 'macro_features'):
                    for macro_name, macro_df in self.macro_features.items():
                        aligned_macro = macro_df.reindex(merged_df.index, method='ffill')
                        for col in aligned_macro.columns:
                            merged_df[f'macro_{col}'] = aligned_macro[col]

                # Add cross-asset features
                if hasattr(self, 'cross_features') and ticker in self.cross_features:
                    cross_df = self.cross_features[ticker]
                    aligned_cross = cross_df.reindex(merged_df.index, method='ffill')
                    for col in aligned_cross.columns:
                        merged_df[f'cross_{col}'] = aligned_cross[col]
                
                # --- THE DEFINITIVE CLEANING PIPELINE ---

                # 1. Ensure the index is sorted chronologically. THIS IS THE KEY FIX.
                merged_df.sort_index(inplace=True)
                
                # 2. Replace infinite values with NaN
                merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # 3. Force all feature columns to be numeric
                feature_cols = [col for col in merged_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
                for col in feature_cols:
                    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
                
                # 4. Use forward-fill and backward-fill. This will now work correctly.
                merged_df.fillna(method='ffill', inplace=True)
                merged_df.fillna(method='bfill', inplace=True)

                # 5. Drop any rows that are still all-NaN (should be very few, if any)
                merged_df.dropna(how='all', inplace=True)

                # 6. Final check: fill any remaining isolated NaNs with 0
                final_nan_count = merged_df.isnull().sum().sum()
                if final_nan_count > 0:
                    print(f"  ⚠️  Found {final_nan_count} isolated NaNs. Filling with 0.")
                    merged_df.fillna(0, inplace=True)
                
                # 7. Final data trim
                min_required_period = 30
                if len(merged_df) > min_required_period:
                    merged_df = merged_df.iloc[min_required_period:]
                
                if len(merged_df) >= 50:
                    merged_data[ticker] = merged_df
                    print(f"  ✅ {ticker}: {merged_df.shape} (rows x features) - SUCCESS. Final NaN count: {merged_df.isnull().sum().sum()}")
                else:
                    print(f"  ❌ {ticker}: Insufficient data after cleaning ({len(merged_df)} rows).")

            except Exception as e:
                print(f"  ❌ Error merging features for {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        print(f"✅ Merged features for {len(merged_data)} tickers")
        return merged_data
      
    def select_causal_features(self, feature_list=None):
        """Select features optimized for causal discovery"""
        if feature_list is None:
            # Comprehensive feature set for causal RL
            feature_list = [
                # Basic price features
                'return', 'log_return', 'volatility', 'momentum_5', 'momentum_10',
                
                # Technical indicators
                'rsi', 'macd', 'bb_position', 'bb_width', 'atr', 'adx',
                
                # Moving averages
                'sma_20', 'sma_50', 'ema_12',
                
                # Volume
                'volume_ratio', 'volume_change',
                
                # Risk measures
                'sharpe_20', 'drawdown',
                
                # Macro features (key ones)
                'macro_sp500_return', 'macro_vix_level', 'macro_vix_change',
                'macro_treasury_10y_return', 'macro_gold_return', 'macro_oil_return',
                'macro_usd_index_return',
                
                # Cross-asset features
                'cross_vix_level', 'cross_vix_regime', 'cross_beta_sp500'
            ]
        
        merged_data = self.merge_all_features()
        
        selected_data = {}
        for ticker, df in merged_data.items():
            # Get available features from the list
            available_features = [f for f in feature_list if f in df.columns]
            
            if len(available_features) < len(feature_list) * 0.7:  # At least 70% of features
                missing = set(feature_list) - set(available_features)
                print(f"  ⚠️  {ticker}: Too many missing features ({len(missing)})")
                print(f"      Missing: {list(missing)[:5]}...")  # Show first 5
                continue
            
            selected_data[ticker] = df[available_features]
            print(f"  ✅ {ticker}: {len(available_features)}/{len(feature_list)} features selected")
        
        self.feature_names = available_features if selected_data else []
        return selected_data
    
    def create_causal_tensor(self, selected_data):
        """Create tensor optimized for causal discovery"""
        print("🔄 Creating causal tensor...")
        
        if not selected_data:
            raise ValueError("No selected data available")
        
        # Get common dates
        all_dates = [df.index for df in selected_data.values()]
        common_dates = all_dates[0]
        for dates in all_dates[1:]:
            common_dates = common_dates.intersection(dates)
        
        print(f"  📅 Common date range: {len(common_dates)} days")
        
        if len(common_dates) < 100:  # Need more data for causal discovery
            print("  ❌ Insufficient common dates for causal analysis")
            raise ValueError("Insufficient common dates for causal analysis")
        
        # Align data
        aligned_data = {}
        for ticker, df in selected_data.items():
            aligned_data[ticker] = df.loc[common_dates].sort_index()
        
        # Create tensor
        tickers = list(aligned_data.keys())
        num_time_steps = len(common_dates)
        num_assets = len(tickers)
        num_features = len(self.feature_names)
        
        data_tensor = np.zeros((num_time_steps, num_assets, num_features))
        
        for i, ticker in enumerate(tickers):
            data_tensor[:, i, :] = aligned_data[ticker].values
        
        # Handle NaN values with more sophisticated interpolation
        if np.any(np.isnan(data_tensor)):
            print("  ⚠️  Cleaning remaining NaN values...")
            for i in range(num_assets):
                for j in range(num_features):
                    series = data_tensor[:, i, j]
                    if np.any(np.isnan(series)):
                        # Use pandas interpolation for better handling
                        series_clean = pd.Series(series).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                        data_tensor[:, i, j] = series_clean.values
        
        print(f"✅ Created causal tensor shape: {data_tensor.shape}")
        print(f"   Time steps: {num_time_steps}")
        print(f"   Assets: {num_assets}")
        print(f"   Features: {num_features}")
        
        return data_tensor, tickers, common_dates
    
    def save_causal_data(self, data_tensor, tickers, dates, save_dir='data'):
        """Save processed causal data with enhanced metadata"""
        # Create the directory structure
        os.makedirs(save_dir, exist_ok=True)
        # Save tensor
        tensor_path = os.path.join(save_dir, 'causal_market_tensor.npy')
        np.save(tensor_path, data_tensor)
        
        # Enhanced metadata for causal analysis
        metadata = {
            'tickers': tickers,
            'feature_names': self.feature_names,
            'dates': dates.strftime('%Y-%m-%d').tolist(),
            'shape': data_tensor.shape,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'macro_indicators': list(self.macro_indicators.keys()),
            'data_quality': {
                'nan_values': int(np.sum(np.isnan(data_tensor))),
                'inf_values': int(np.sum(np.isinf(data_tensor))),
                'data_range': [float(np.min(data_tensor)), float(np.max(data_tensor))],
                'num_time_steps': int(data_tensor.shape[0]),
                'num_assets': int(data_tensor.shape[1]),
                'num_features': int(data_tensor.shape[2])
            },
            'feature_categories': self._categorize_features()
        }
        
        metadata_path = os.path.join(save_dir, 'causal_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature mapping
        features_path = os.path.join(save_dir, 'causal_features.txt')
        with open(features_path, 'w') as f:
            f.write("CAUSAL FEATURE MAPPING\n")
            f.write("=" * 50 + "\n\n")
            
            categories = self._categorize_features()
            for category, features in categories.items():
                f.write(f"{category.upper()}:\n")
                for i, feature in enumerate(features):
                    if feature in self.feature_names:
                        idx = self.feature_names.index(feature)
                        f.write(f"  [{idx:2d}] {feature}\n")
                f.write("\n")
        
        # Save raw data for causal discovery algorithms
        raw_data_path = os.path.join(save_dir, 'raw_aligned_data.pkl')
        import pickle
        
        aligned_raw_data = {}
        for i, ticker in enumerate(tickers):
            if ticker in self.processed_data:
                aligned_raw_data[ticker] = self.processed_data[ticker].loc[dates]
        
        with open(raw_data_path, 'wb') as f:
            pickle.dump(aligned_raw_data, f)
        
        print(f"✅ Saved causal data to {save_dir}/")
        print(f"   - Tensor: {tensor_path}")
        print(f"   - Metadata: {metadata_path}")
        print(f"   - Features: {features_path}")
        print(f"   - Raw data: {raw_data_path}")
        
        return tensor_path, metadata_path
    
    def _categorize_features(self):
        """Categorize features for better understanding"""
        categories = {
            'price_momentum': ['return', 'log_return', 'momentum_5', 'momentum_10', 'momentum_20'],
            'technical_indicators': ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position', 'bb_width', 'atr', 'adx'],
            'moving_averages': ['sma_20', 'sma_50', 'ema_12'],
            'volatility_risk': ['volatility', 'volatility_long', 'sharpe_20', 'drawdown'],
            'volume': ['volume_ratio', 'volume_change'],
            'macro_market': [f for f in self.feature_names if f.startswith('macro_')],
            'cross_asset': [f for f in self.feature_names if f.startswith('cross_')],
            'price_ratios': ['high_low_ratio', 'close_to_high', 'close_to_low']
        }
        
        # Filter to only include features that exist
        filtered_categories = {}
        for category, features in categories.items():
            existing_features = [f for f in features if f in self.feature_names]
            if existing_features:
                filtered_categories[category] = existing_features
        
        return filtered_categories
    
    def run_causal_pipeline(self, feature_list=None, save_dir='data', use_synthetic=False):
        """Run the complete causal data collection pipeline"""
        print("🚀 Starting causal portfolio data collection pipeline...")
        print(f"📊 Target assets: {len(self.tickers)}")
        print(f"🌍 Macro indicators: {len(self.macro_indicators)}")
        print(f"💾 Save directory: {save_dir}")
        
        # Step 1: Download stock data
        if not use_synthetic:
            print("\n=== STEP 1: Stock Data ===")
            self.download_stock_data()
            
            if not self.raw_data:
                print("❌ No stock data downloaded, switching to synthetic...")
                use_synthetic = True
        
        if use_synthetic:
            print("\n=== SYNTHETIC DATA GENERATION ===")
            self.generate_synthetic_data()
            self.generate_synthetic_macro_data()
        else:
            # Step 2: Download macro data
            print("\n=== STEP 2: Macro Data ===")
            self.download_macro_data()
            
            if not self.macro_data:
                print("❌ No macro data downloaded, generating synthetic...")
                self.generate_synthetic_macro_data()
        
        # Step 3: Calculate all features
        print("\n=== STEP 3: Feature Engineering ===")
        
        # Stock features
        print("📈 Processing stock features...")
        self.calculate_enhanced_stock_features()
        
        # Macro features
        print("🌍 Processing macro features...")
        self.calculate_macro_features()
        
        # Cross-asset features
        print("🔄 Processing cross-asset features...")
        self.calculate_cross_asset_features()
        
        # Step 4: Select and merge features
        print("\n=== STEP 4: Feature Selection & Merging ===")
        selected_data = self.select_causal_features(feature_list)
        
        if not selected_data:
            raise ValueError("No data survived feature selection process")
        
        # Step 5: Create causal tensor
        print("\n=== STEP 5: Tensor Creation ===")
        data_tensor, tickers, dates = self.create_causal_tensor(selected_data)
        
        # Step 6: Save everything
        print("\n=== STEP 6: Data Persistence ===")
        tensor_path, metadata_path = self.save_causal_data(data_tensor, tickers, dates, save_dir)
        
        # Step 7: Quality assessment
        print("\n=== STEP 7: Quality Assessment ===")
        self._assess_data_quality(data_tensor, tickers)
        
        print("\n🎉 Causal pipeline completed successfully!")
        return data_tensor, tickers, dates
    
    def _assess_data_quality(self, data_tensor, tickers):
        """Assess data quality for causal analysis"""
        print("🔍 Data Quality Assessment:")
        
        # Basic statistics
        print(f"   ✓ Shape: {data_tensor.shape}")
        print(f"   ✓ Date range: {len(data_tensor)} time steps")
        print(f"   ✓ Assets: {len(tickers)}")
        print(f"   ✓ Features: {len(self.feature_names)}")
        
        # Data quality metrics
        nan_count = np.sum(np.isnan(data_tensor))
        inf_count = np.sum(np.isinf(data_tensor))
        
        print(f"   ✓ NaN values: {nan_count} ({nan_count/data_tensor.size*100:.2f}%)")
        print(f"   ✓ Infinite values: {inf_count}")
        print(f"   ✓ Value range: [{np.min(data_tensor):.4f}, {np.max(data_tensor):.4f}]")
        
        # Feature diversity check
        feature_vars = []
        for i in range(data_tensor.shape[2]):
            var = np.var(data_tensor[:, :, i])
            feature_vars.append(var)
        
        zero_var_features = sum(1 for var in feature_vars if var < 1e-10)
        print(f"   ✓ Zero-variance features: {zero_var_features}/{len(feature_vars)}")
        
        if zero_var_features > 0:
            print("   ⚠️  Some features have very low variance - consider removal")
        
        # Temporal coverage
        coverage_ratio = data_tensor.shape[0] / 252  # Assuming daily data, ~252 trading days/year
        print(f"   ✓ Temporal coverage: {coverage_ratio:.1f} years")
        
        if coverage_ratio < 2:
            print("   ⚠️  Less than 2 years of data - may limit causal discovery")
        
        print("✅ Quality assessment complete")
    
    def generate_synthetic_data(self):
        """Generate synthetic stock data"""
        print("🔧 Generating synthetic stock data...")
        
        np.random.seed(42)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        dates = dates[dates.dayofweek < 5]  # Remove weekends
        # Ensure timezone-naive
        dates = dates.tz_localize(None) if dates.tz is not None else dates
        
        for ticker in self.tickers:
            print(f"  Generating data for {ticker}...")
            
            n_days = len(dates)
            base_price = np.random.uniform(50, 300)
            returns = np.random.normal(0.0005, 0.02, n_days)
            
            # Add some cross-correlations for realism
            if ticker != self.tickers[0]:  # Add correlation with first stock
                first_ticker_returns = np.random.normal(0.0005, 0.02, n_days)
                correlation = np.random.uniform(0.3, 0.7)
                returns = correlation * first_ticker_returns + np.sqrt(1 - correlation**2) * returns
            
            trend = np.random.normal(0, 0.0001, n_days).cumsum()
            returns += trend
            
            prices = base_price * np.exp(returns.cumsum())
            
            volume_base = np.random.uniform(1000000, 10000000)
            volume = volume_base * (1 + 0.5 * np.abs(returns) + np.random.normal(0, 0.1, n_days))
            volume = np.maximum(volume, 1000)
            
            high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
            low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
            open_price = np.roll(prices, 1)
            open_price[0] = prices[0]
            
            df = pd.DataFrame({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': prices,
                'Volume': volume
            }, index=dates[:n_days])
            
            self.raw_data[ticker] = df
            print(f"  ✅ {ticker}: {len(df)} synthetic rows")
        
        print(f"✅ Generated synthetic stock data for {len(self.raw_data)} tickers")


def load_causal_data(data_dir='data'):
    """Load saved causal portfolio data"""
    tensor_path = os.path.join(data_dir, 'causal_market_tensor.npy')
    metadata_path = os.path.join(data_dir, 'causal_metadata.json')
    
    if not os.path.exists(tensor_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Causal data not found in {data_dir}")
    
    data_tensor = np.load(tensor_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return data_tensor, metadata


# Example usage for causal RL
if __name__ == "__main__":
    # Enhanced ticker set for better causal relationships
    tickers = [
        # Large Cap Tech (high correlation expected)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        
        # Financial sector (interest rate sensitive)
        'JPM', 'BAC', 'GS', 'WFC',
        
        # Energy (oil price sensitive)
        'XOM', 'CVX', 'COP',
        
        # Defensive sectors
        'PG', 'JNJ', 'KO', 'UNH',
        
        # Cyclical sectors
        'CAT', 'HD', 'BA',
        
        # Market benchmark
        'SPY'
    ]
    
    # Initialize causal collector
    collector = CausalPortfolioDataCollector(
        tickers=tickers,
        start_date='2018-01-01',
        end_date='2024-01-01'
    )
    
    # Custom feature selection for causal discovery
    causal_features = [
        # Core price features
        'return', 'log_return', 'volatility', 'momentum_10',
        
        # Technical signals
        'rsi', 'macd', 'bb_position', 'atr',
        
        # Market regime
        'macro_sp500_return', 'macro_vix_level', 'macro_vix_change',
        
        # Economic factors
        'macro_treasury_10y_return', 'macro_gold_return', 'macro_oil_return',
        'macro_usd_index_return',
        
        # Cross-asset effects
        'cross_vix_regime', 'cross_beta_sp500',
        
        # Sector rotation
        'macro_financials_momentum', 'macro_technology_momentum', 'macro_energy_momentum'
    ]
    
    try:
        print("🔄 Running causal data pipeline...")
        data_tensor, final_tickers, dates = collector.run_causal_pipeline(
            feature_list=causal_features,
            save_dir='data',  # Changed default save directory
            use_synthetic=False  # Try real data first
        )
        
        print(f"\n📊 Causal Dataset Ready:")
        print(f"   Shape: {data_tensor.shape}")
        print(f"   Tickers: {len(final_tickers)}")
        print(f"   Features: {len(collector.feature_names)}")
        print(f"   Time span: {len(dates)} days")
        print(f"   Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        
        # Feature breakdown
        categories = collector._categorize_features()
        print(f"\n📋 Feature Categories:")
        for category, features in categories.items():
            print(f"   {category}: {len(features)} features")
        
        print(f"\n✅ Data ready for causal RL algorithms!")
        print(f"   💾 Saved to: data/")
        print(f"   🔧 Use load_causal_data('data') to reload")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        print("🔄 Trying with synthetic data...")
        
        try:
            data_tensor, final_tickers, dates = collector.run_causal_pipeline(
                feature_list=causal_features,
                save_dir='data',  # Keep same directory for synthetic data
                use_synthetic=True
            )
            print(f"\n✅ Synthetic causal data ready: {data_tensor.shape}")
            print(f"   ⚠️  Remember: This is synthetic data for testing")
            
        except Exception as e2:
            print(f"❌ Synthetic generation also failed: {str(e2)}")
    
    # Test loading
    print(f"\n📂 Testing data loading...")
    try:
        loaded_tensor, metadata = load_causal_data('data')
        print(f"✅ Successfully loaded causal data: {loaded_tensor.shape}")
        print(f"📊 Available features: {len(metadata['feature_names'])}")
        print(f"🏢 Assets: {len(metadata['tickers'])}")
        print(f"🌍 Macro indicators: {len(metadata['macro_indicators'])}")
        
    except FileNotFoundError:
        print(f"📂 Causal data not found, try running pipeline first")
    except Exception as e:
        print(f"❌ Loading error: {str(e)}")