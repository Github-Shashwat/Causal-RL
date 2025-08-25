import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Causal-learn imports
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam import ICALiNGAM, DirectLiNGAM
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.Granger.Granger import Granger
from causallearn.search.ScoreBased.GES import ges
from statsmodels.tsa.stattools import grangercausalitytests

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from causal_data_loader import CausalPortfolioDataCollector, load_causal_data

import json
import os
import pickle
from datetime import datetime

class FinancialCausalGraphBuilder:
    """
    Causal graph builder specifically designed for financial market data
    Integrates with CausalPortfolioDataCollector and uses multiple causal discovery algorithms
    AUTO-SAVES ALL VISUALIZATIONS - Colab-friendly version
    """
    
    def __init__(self, data_source=None, data_dir='data', load_existing=True, auto_save=True):
        """
        Initialize the causal graph builder
        
        Args:
            data_source: CausalPortfolioDataCollector instance or None to load from disk
            data_dir: Directory containing saved causal data
            load_existing: Whether to load existing data from disk
            auto_save: Whether to automatically save all visualizations
        """
        self.data_dir = data_dir
        self.auto_save = auto_save
        self.graphs = {}
        self.results = {}
        self.data_tensor = None
        self.tickers = None
        self.dates = None
        self.feature_names = None
        self.metadata = None
        
        # Set up visualization directories
        self.viz_dir = os.path.join(data_dir, 'visualizations')
        if auto_save:
            self._setup_viz_directories()
        
        # Financial domain knowledge
        self.domain_knowledge = self._define_financial_domain_knowledge()
        
        if load_existing and os.path.exists(os.path.join(data_dir, 'causal_market_tensor.npy')):
            self.load_data()
        elif data_source is not None:
            self.load_from_collector(data_source)
        else:
            print("⚠️  No data provided. Use load_data() or load_from_collector() first.")
    
    def _setup_viz_directories(self):
        """Setup visualization directory structure"""
        dirs_to_create = [
            self.viz_dir,
            os.path.join(self.viz_dir, 'individual_graphs'),
            os.path.join(self.viz_dir, 'algorithm_comparisons'),
            os.path.join(self.viz_dir, 'mode_comparisons'),
            os.path.join(self.viz_dir, 'summary_reports'),
            os.path.join(self.viz_dir, 'cross_asset')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        if self.auto_save:
            print(f"📁 Visualization directories created in: {self.viz_dir}")
    
    def _get_save_path(self, category: str, filename: str) -> str:
        """Generate save path for visualization files"""
        if not self.auto_save:
            return None
        return os.path.join(self.viz_dir, category, filename)
    
    def _define_financial_domain_knowledge(self) -> Dict:
        """Define financial domain knowledge for causal discovery"""
        return {
            'forbidden_edges': [
                # Future cannot cause past (temporal constraints will be handled separately)
                # Volume typically doesn't directly cause macro indicators
                ('volume_ratio', 'macro_sp500_return'),
                ('volume_change', 'macro_treasury_10y_return'),
                
                # Individual stock features shouldn't cause broad market indices
                ('rsi', 'macro_sp500_return'),
                ('bb_position', 'macro_vix_level'),
            ],
            
            'encouraged_edges': [
                # Market → Individual stocks (strong evidence)
                ('macro_sp500_return', 'return'),
                ('macro_vix_level', 'volatility'),
                ('macro_vix_change', 'return'),
                
                # Interest rates → Financial stocks
                ('macro_treasury_10y_return', 'return'),
                
                # Oil → Energy stocks (if energy stocks present)
                ('macro_oil_return', 'return'),
                
                # Cross-asset relationships
                ('cross_vix_level', 'volatility'),
                ('cross_beta_sp500', 'return'),
            ],
            
            'temporal_lags': {
                # Macro indicators typically lead individual stocks
                'macro_indicators': 1,  # 1-day lag
                'cross_asset': 0,       # Same day
                'technical': 0,         # Same day
            },
            
            'feature_categories': {
                'macro': ['macro_sp500_return', 'macro_vix_level', 'macro_vix_change', 
                         'macro_treasury_10y_return', 'macro_gold_return', 'macro_oil_return'],
                'technical': ['rsi', 'macd', 'bb_position', 'bb_width', 'atr', 'adx'],
                'momentum': ['return', 'log_return', 'momentum_5', 'momentum_10'],
                'volatility': ['volatility', 'volatility_long'],
                'volume': ['volume_ratio', 'volume_change'],
                'cross_asset': ['cross_vix_level', 'cross_vix_regime', 'cross_beta_sp500']
            }
        }
    
    def load_data(self):
        """Load causal data from disk"""
        try:
            self.data_tensor, self.metadata = load_causal_data(self.data_dir)
            self.tickers = self.metadata['tickers']
            self.feature_names = self.metadata['feature_names']
            self.dates = pd.to_datetime(self.metadata['dates'])
            
            print(f"✅ Loaded causal data:")
            print(f"   Shape: {self.data_tensor.shape}")
            print(f"   Tickers: {len(self.tickers)}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Date range: {self.dates[0].strftime('%Y-%m-%d')} to {self.dates[-1].strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"❌ Failed to load causal data: {e}")
            raise
    
    def load_from_collector(self, collector: CausalPortfolioDataCollector):
        """Load data directly from CausalPortfolioDataCollector"""
        if collector.processed_data is None:
            raise ValueError("Collector has no processed data. Run the pipeline first.")
        
        # This would require the collector to have run the pipeline
        # For now, we'll assume data is saved and load from disk
        self.load_data()
    
    def prepare_asset_data(self, ticker: str, max_features: int = 20, force_log_return: bool = True):
        """
        Prepare data for a specific asset for causal discovery.
        - If force_log_return=True, always include 'log_return'.
        - If False, select features purely by importance.
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} not found in data")
        
        ticker_idx = self.tickers.index(ticker)
        asset_data = self.data_tensor[:, ticker_idx, :]
        
        # Find the index of our target variable, 'log_return'
        log_return_idx = -1
        if 'log_return' in self.feature_names:
            log_return_idx = self.feature_names.index('log_return')

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(asset_data)
        
        if force_log_return and log_return_idx != -1:
            # Prevent log_return from being chosen in top features automatically
            feature_importance[log_return_idx] = -np.inf
            top_feature_indices = np.argsort(feature_importance)[-(max_features - 1):]
            final_indices = np.concatenate([[log_return_idx], top_feature_indices])
        else:
            # Normal feature selection
            top_feature_indices = np.argsort(feature_importance)[-max_features:]
            final_indices = top_feature_indices

        selected_data = asset_data[:, final_indices]
        selected_feature_names = [self.feature_names[i] for i in final_indices]
        
        mode = "FORCED log_return" if force_log_return else "FREE selection"
        print(f"📊 {ticker}: Selected {len(selected_feature_names)} features for causal discovery ({mode})")
        print(f"   Features: {selected_feature_names}")
        
        return selected_data, selected_feature_names, final_indices
    
    def _calculate_feature_importance(self, data: np.ndarray) -> np.ndarray:
        """Calculate feature importance based on variance and non-stationarity"""
        importance = np.zeros(data.shape[1])
        
        for i in range(data.shape[1]):
            series = data[:, i]
            
            # Variance component
            var_score = np.var(series)
            
            # Non-stationarity score (using differencing)
            diff_var = np.var(np.diff(series))
            
            # Combine scores
            importance[i] = var_score * (1 + diff_var)
        
        return importance
    
    def create_background_knowledge(self, feature_names: List[str]) -> BackgroundKnowledge:
        """Create background knowledge constraints for financial data"""
        bk = BackgroundKnowledge()
        
        # Map feature names to their integer indices
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        # Add forbidden edges based on financial domain knowledge
        for source, target in self.domain_knowledge['forbidden_edges']:
            if source in feature_to_idx and target in feature_to_idx:
                # Get the integer index for the source and target nodes
                source_idx = feature_to_idx[source]
                target_idx = feature_to_idx[target]
                # Pass the integer indices to the function
                bk.add_forbidden_by_node(source_idx, target_idx)
                
        # Add encouraged relationships (optional, and also using indices)
        encouraged_count = 0
        for source, target in self.domain_knowledge['encouraged_edges'][:3]:
            if source in feature_to_idx and target in feature_to_idx:
                # source_idx = feature_to_idx[source]
                # target_idx = feature_to_idx[target]
                # bk.add_required_by_node(source_idx, target_idx)
                encouraged_count += 1
                
        print(f"🧠 Applied domain knowledge:")
        print(f"   Forbidden edges applied: {len(bk.forbidden_rules)}")
        print(f"   Encouraged edges considered: {encouraged_count}")
        
        return bk
    
    def run_pc_algorithm(self, data: np.ndarray, feature_names: List[str], 
                        alpha: float = 0.05, use_background_knowledge: bool = True) -> Dict:
        """
        Run PC algorithm for causal discovery
        """
        print(f"🔍 Running PC algorithm (alpha={alpha})...")
        
        try:
            cg = pc(data, alpha=alpha, indep_test='fisherz', 
                  stable=True, background_knowledge=None,
                  verbose=False, show_progress=True)
            
            # Convert the CausalLearn graph object to a NumPy adjacency matrix
            adj_matrix = cg.G.graph

            result = {
                'algorithm': 'PC',
                'graph': adj_matrix, # Return the NumPy array
                'alpha': alpha,
                'num_edges': np.sum(adj_matrix != 0),
                'feature_names': feature_names,
                'background_knowledge': False
            }
            
            print(f"✅ PC completed: {result['num_edges']} edges found")
            return result
            
        except Exception as e:
            print(f"❌ PC algorithm failed: {e}")
            return None
    
    def run_ges_algorithm(self, data: np.ndarray, feature_names: List[str],
                        score_func: str = 'local_score_BIC') -> Dict:
        """
        Run GES algorithm for causal discovery
        """
        print(f"🔍 Running GES algorithm (score={score_func})...")
        
        try:
            # Run GES algorithm
            record = ges(data, score_func=score_func)
            
            # Convert the CausalLearn graph object to a NumPy adjacency matrix
            adj_matrix = record['G'].graph

            result = {
                'algorithm': 'GES',
                'graph': adj_matrix, # Return the NumPy array
                'score_func': score_func,
                'num_edges': np.sum(adj_matrix != 0),
                'feature_names': feature_names,
                'score': record.get('score', None)
            }
            
            print(f"✅ GES completed: {result['num_edges']} edges found")
            return result
            
        except Exception as e:
            print(f"❌ GES algorithm failed: {e}")
            return None
    
    def run_lingam_algorithm(self, data: np.ndarray, feature_names: List[str],
                            method: str = 'ICALiNGAM') -> Dict:
        """
        Run LiNGAM algorithm for linear causal discovery
        
        Args:
            data: Input data (time_steps, features)
            feature_names: List of feature names
            method: LiNGAM method ('ICALiNGAM' or 'DirectLiNGAM')
        
        Returns:
            Dictionary with results
        """
        print(f"🔍 Running {method} algorithm...")
        
        try:
            if method == 'ICALiNGAM':
                model = ICALiNGAM()
            elif method == 'DirectLiNGAM':
                model = DirectLiNGAM()
            else:
                raise ValueError(f"Unknown LiNGAM method: {method}")
            
            model.fit(data)
            
            result = {
                'algorithm': method,
                'adjacency_matrix': model.adjacency_matrix_,
                'causal_order': model.causal_order_,
                'num_edges': np.sum(model.adjacency_matrix_ != 0),
                'feature_names': feature_names
            }
            
            print(f"✅ {method} completed: {result['num_edges']} edges found")
            return result
            
        except Exception as e:
            print(f"❌ {method} algorithm failed: {e}")
            return None
    
    def run_granger_causality(self, data: np.ndarray, feature_names: list, maxlag: int = 5) -> dict:
        """
        Run Granger causality test using statsmodels
        Returns adjacency-like matrix with p-values
        """
        print(f"🔍 Running Granger causality (maxlag={maxlag})...")

        try:
            n_features = data.shape[1]
            granger_matrix = np.ones((n_features, n_features))  # store p-values

            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        continue
                    # Test if j → i (does j Granger-cause i?)
                    test_result = grangercausalitytests(
                        data[:, [i, j]], maxlag=maxlag, verbose=False
                    )
                    # take the min p-value across lags
                    p_values = [res[0]["ssr_ftest"][1] for lag, res in test_result.items()]
                    granger_matrix[i, j] = min(p_values)

            result = {
                "algorithm": "Granger",
                "granger_matrix": granger_matrix,
                "maxlag": maxlag,
                "feature_names": feature_names,
                "num_edges": int(np.sum(granger_matrix < 0.05))  # significant edges at 5%
            }

            print(f"✅ Granger causality completed: {result['num_edges']} significant relationships")
            return result

        except Exception as e:
            print(f"❌ Granger causality failed: {e}")
            return None
    
    def discover_causal_graph(self, ticker: str, algorithms: List[str] = None,
                             max_features: int = 15, force_log_return: bool = True) -> Dict:
        """
        Discover causal graph for a specific asset using multiple algorithms
        
        Args:
            ticker: Asset ticker
            algorithms: List of algorithms to run
            max_features: Maximum number of features
            force_log_return: Whether to force include log_return
        
        Returns:
            Dictionary containing all results
        """
        if algorithms is None:
            algorithms = ['PC', 'GES', 'ICALiNGAM', 'Granger']
        
        mode_text = "FORCED log_return" if force_log_return else "FREE selection"
        print(f"\n🎯 Discovering causal graph for {ticker} ({mode_text})")
        print(f"   Algorithms: {algorithms}")
        print(f"   Max features: {max_features}")
        
        # Prepare data
        data, feature_names, feature_indices = self.prepare_asset_data(ticker, max_features, force_log_return)
        
        # Store data info
        asset_results = {
            'ticker': ticker,
            'mode': 'forced' if force_log_return else 'free',
            'data_shape': data.shape,
            'feature_names': feature_names,
            'feature_indices': feature_indices,
            'algorithms': {}
        }
        
        # Run each algorithm
        for algorithm in algorithms:
            if algorithm == 'PC':
                result = self.run_pc_algorithm(data, feature_names)
            elif algorithm == 'GES':
                result = self.run_ges_algorithm(data, feature_names)
            elif algorithm == 'ICALiNGAM':
                result = self.run_lingam_algorithm(data, feature_names, 'ICALiNGAM')
            elif algorithm == 'DirectLiNGAM':
                result = self.run_lingam_algorithm(data, feature_names, 'DirectLiNGAM')
            elif algorithm == 'Granger':
                result = self.run_granger_causality(data, feature_names)
            else:
                print(f"⚠️  Unknown algorithm: {algorithm}")
                continue
            
            if result is not None:
                asset_results['algorithms'][algorithm] = result
        
        # Create unique key for results storage
        result_key = f"{ticker}_{asset_results['mode']}"
        
        # Store results
        self.results[result_key] = asset_results
        
        print(f"✅ Causal discovery completed for {ticker} ({mode_text})")
        return asset_results
    
    def visualize_causal_graph(self, ticker: str, algorithm: str = 'PC', mode: str = 'forced',
                              threshold: float = 0.01, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Visualize causal graph for a specific ticker and algorithm
        AUTO-SAVES to individual_graphs/ folder
        """
        result_key = f"{ticker}_{mode}"
        
        if result_key not in self.results:
            print(f"❌ No results for {ticker} in {mode} mode. Run discover_causal_graph first.")
            return
        
        if algorithm not in self.results[result_key]['algorithms']:
            print(f"❌ No {algorithm} results for {ticker} in {mode} mode")
            return
        
        result = self.results[result_key]['algorithms'][algorithm]
        feature_names = result['feature_names']
        
        # Get the adjacency matrix from the result
        if algorithm in ['PC', 'GES']:
            adj_matrix = result['graph']
        elif 'LiNGAM' in algorithm:
            adj_matrix = result['adjacency_matrix']
        elif algorithm == 'Granger':
            granger_matrix = result['granger_matrix']
            adj_matrix = (granger_matrix < 0.05).astype(float) # Convert p-values to a binary graph
        else:
            print(f"❌ Unknown algorithm for visualization: {algorithm}")
            return

        # Create a NetworkX DiGraph from the adjacency matrix
        G = nx.DiGraph(adj_matrix)
        
        # Relabel nodes with their proper feature names
        labels = {i: name for i, name in enumerate(feature_names)}
        nx.relabel_nodes(G, labels, copy=False)

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42) # Layout for the nodes
        
        # Get node colors
        node_colors = self._get_node_colors(feature_names)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray', arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

        mode_text = "Forced log_return" if mode == 'forced' else "Free selection"
        plt.title(f'Causal Graph: {ticker} ({algorithm}) - {mode_text}', size=16, weight='bold')
        plt.axis('off')
        
        # Auto-save
        save_path = self._get_save_path('individual_graphs', f'{ticker}_{mode}_{algorithm}.png')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Graph saved to {save_path}")
        
        plt.show()
        
    def _get_node_colors(self, feature_names: List[str]) -> List[str]:
        """Get node colors based on feature categories"""
        colors = []
        color_map = {
            'macro': '#FF6B6B',      # Red
            'technical': '#4ECDC4',   # Teal
            'momentum': '#45B7D1',    # Blue
            'volatility': '#96CEB4',  # Green
            'volume': '#FFEAA7',      # Yellow
            'cross_asset': '#DDA0DD',  # Plum
            'default': '#95A5A6'       # Gray
        }
        
        for name in feature_names:
            category = 'default'
            for cat, features in self.domain_knowledge['feature_categories'].items():
                if any(feat in name for feat in features):
                    category = cat
                    break
            colors.append(color_map.get(category, color_map['default']))
        
        return colors
    
    def _add_graph_legend(self):
        """Add legend to graph visualization"""
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=2, label='Positive Effect'),
            plt.Line2D([0], [0], color='red', lw=2, label='Negative Effect'),
            plt.scatter([], [], c='#FF6B6B', s=100, label='Macro Features'),
            plt.scatter([], [], c='#4ECDC4', s=100, label='Technical Features'),
            plt.scatter([], [], c='#45B7D1', s=100, label='Momentum Features'),
            plt.scatter([], [], c='#96CEB4', s=100, label='Volatility Features'),
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    def compare_algorithms(self, ticker: str, mode: str = 'forced') -> None:
        """
        Compare results from different algorithms for a ticker
        AUTO-SAVES to algorithm_comparisons/ folder
        """
        result_key = f"{ticker}_{mode}"
        
        if result_key not in self.results:
            print(f"❌ No results for {ticker} in {mode} mode")
            return
        
        algorithms = list(self.results[result_key]['algorithms'].keys())
        if len(algorithms) < 2:
            print(f"⚠️  Need at least 2 algorithms for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, algorithm in enumerate(algorithms[:4]):  # Max 4 algorithms
            if i >= len(axes):
                break
                
            ax = axes[i]
            result = self.results[result_key]['algorithms'][algorithm]
            feature_names = result['feature_names']
            
            # Create adjacency matrix for visualization
            if algorithm in ['PC', 'GES']:
                adj_matrix = result['graph']
            elif 'LiNGAM' in algorithm:
                adj_matrix = result['adjacency_matrix']
            elif algorithm == 'Granger':
                granger_matrix = result['granger_matrix']
                adj_matrix = (granger_matrix < 0.05).astype(float)
            
            # Plot as heatmap
            im = ax.imshow(adj_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            ax.set_xticks(range(len(feature_names)))
            ax.set_yticks(range(len(feature_names)))
            ax.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                               for name in feature_names], rotation=45, ha='right')
            ax.set_yticklabels([name[:10] + '...' if len(name) > 10 else name 
                               for name in feature_names])
            
            ax.set_title(f'{algorithm}\n{result.get("num_edges", 0)} edges', 
                        weight='bold', pad=10)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide unused subplots
        for i in range(len(algorithms), len(axes)):
            axes[i].set_visible(False)
        
        mode_text = "Forced log_return" if mode == 'forced' else "Free selection"
        plt.suptitle(f'Causal Discovery Comparison: {ticker} - {mode_text}', size=16, weight='bold')
        plt.tight_layout()
        
        # Auto-save
        save_path = self._get_save_path('algorithm_comparisons', f'{ticker}_{mode}_comparison.png')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comparison saved to {save_path}")
        
        plt.show()
    
    def compare_modes(self, ticker: str, algorithm: str = 'GES') -> None:
        """
        Compare forced vs free mode for a specific ticker and algorithm
        AUTO-SAVES to mode_comparisons/ folder
        """
        forced_key = f"{ticker}_forced"
        free_key = f"{ticker}_free"
        
        if forced_key not in self.results or free_key not in self.results:
            print(f"❌ Need both forced and free results for {ticker}")
            return
        
        if algorithm not in self.results[forced_key]['algorithms'] or algorithm not in self.results[free_key]['algorithms']:
            print(f"❌ Algorithm {algorithm} not found in both modes")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        for ax, mode, result_key in [(ax1, 'Forced', forced_key), (ax2, 'Free', free_key)]:
            result = self.results[result_key]['algorithms'][algorithm]
            feature_names = result['feature_names']
            
            # Get adjacency matrix
            if algorithm in ['PC', 'GES']:
                adj_matrix = result['graph']
            elif 'LiNGAM' in algorithm:
                adj_matrix = result['adjacency_matrix']
            elif algorithm == 'Granger':
                granger_matrix = result['granger_matrix']
                adj_matrix = (granger_matrix < 0.05).astype(float)
            
            # Plot as heatmap
            im = ax.imshow(adj_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            ax.set_xticks(range(len(feature_names)))
            ax.set_yticks(range(len(feature_names)))
            ax.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                               for name in feature_names], rotation=45, ha='right')
            ax.set_yticklabels([name[:10] + '...' if len(name) > 10 else name 
                               for name in feature_names])
            
            ax.set_title(f'{mode} Mode\n{result.get("num_edges", 0)} edges', 
                        weight='bold', pad=10)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.suptitle(f'Mode Comparison: {ticker} ({algorithm})', size=16, weight='bold')
        plt.tight_layout()
        
        # Auto-save
        save_path = self._get_save_path('mode_comparisons', f'{ticker}_{algorithm}_mode_comparison.png')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Mode comparison saved to {save_path}")
        
        plt.show()
    
    def analyze_cross_asset_causality(self, algorithms: List[str] = None,
                                    max_features: int = 10, run_both_modes: bool = True) -> Dict:
        """
        Analyze causality relationships across multiple assets
        AUTO-SAVES cross-asset visualizations
        
        Args:
            algorithms: Algorithms to use
            max_features: Maximum features per asset
            run_both_modes: Whether to run both forced and free modes
        
        Returns:
            Cross-asset causality analysis results
        """
        if algorithms is None:
            algorithms = ['GES', 'Granger']  # Use faster algorithms for cross-asset
        
        print(f"\n🌍 Cross-Asset Causality Analysis")
        print(f"   Assets: {len(self.tickers)}")
        print(f"   Algorithms: {algorithms}")
        print(f"   Both modes: {run_both_modes}")
        
        cross_results = {}
        
        # Determine which modes to run
        modes_to_run = [True, False] if run_both_modes else [True]
        
        # Run analysis for each asset and mode
        for ticker in self.tickers:  # Limit to first 10 for demo
            print(f"\n📊 Processing {ticker}...")
            
            for force_mode in modes_to_run:
                mode_name = "forced" if force_mode else "free"
                print(f"   Running {mode_name} mode...")
                
                asset_result = self.discover_causal_graph(
                    ticker, algorithms, max_features, force_mode
                )
                
                result_key = f"{ticker}_{mode_name}"
                cross_results[result_key] = asset_result
        
        # Aggregate results
        aggregated = self._aggregate_cross_asset_results(cross_results)
        
        # Create cross-asset visualizations
        self._create_cross_asset_visualizations(aggregated, run_both_modes)
        
        return aggregated
    
    def _aggregate_cross_asset_results(self, cross_results: Dict) -> Dict:
        """Aggregate cross-asset causality results"""
        aggregated = {
            'common_patterns': {},
            'asset_specific': {},
            'macro_influence': {},
            'cross_correlations': {},
            'mode_comparison': {}
        }
        
        # Find common causal patterns
        all_edges = {}
        
        for result_key, result in cross_results.items():
            ticker, mode = result_key.split('_', 1)
            
            for algorithm, algo_result in result['algorithms'].items():
                algo_mode_key = f"{algorithm}_{mode}"
                if algo_mode_key not in all_edges:
                    all_edges[algo_mode_key] = {}
                
                feature_names = algo_result['feature_names']
                
                # Extract edges
                if algorithm in ['PC', 'GES']:
                    adj_matrix = algo_result['graph']
                elif 'LiNGAM' in algorithm:
                    adj_matrix = algo_result['adjacency_matrix']
                elif algorithm == 'Granger':
                    granger_matrix = algo_result['granger_matrix']
                    adj_matrix = (granger_matrix < 0.05).astype(float)
                
                # Store edges
                for i in range(len(feature_names)):
                    for j in range(len(feature_names)):
                        if adj_matrix[i, j] != 0:
                            edge = f"{feature_names[i]} -> {feature_names[j]}"
                            if edge not in all_edges[algo_mode_key]:
                                all_edges[algo_mode_key][edge] = []
                            all_edges[algo_mode_key][edge].append(ticker)
        
        # Find common patterns
        for algo_mode_key, edges in all_edges.items():
            common_edges = {edge: assets for edge, assets in edges.items() 
                           if len(assets) >= max(2, len(set(k.split('_')[0] for k in cross_results.keys())) * 0.3)}
            aggregated['common_patterns'][algo_mode_key] = common_edges
        
        # Compare modes if both are available
        forced_patterns = {}
        free_patterns = {}
        
        for algo_mode_key in all_edges.keys():
            if 'forced' in algo_mode_key:
                algo = algo_mode_key.replace('_forced', '')
                forced_patterns[algo] = all_edges[algo_mode_key]
            elif 'free' in algo_mode_key:
                algo = algo_mode_key.replace('_free', '')
                free_patterns[algo] = all_edges[algo_mode_key]
        
        # Analyze differences between modes
        mode_differences = {}
        for algo in forced_patterns.keys():
            if algo in free_patterns:
                forced_edges = set(forced_patterns[algo].keys())
                free_edges = set(free_patterns[algo].keys())
                
                mode_differences[algo] = {
                    'forced_only': forced_edges - free_edges,
                    'free_only': free_edges - forced_edges,
                    'common': forced_edges & free_edges,
                    'forced_total': len(forced_edges),
                    'free_total': len(free_edges),
                    'overlap_ratio': len(forced_edges & free_edges) / max(1, len(forced_edges | free_edges))
                }
        
        aggregated['mode_comparison'] = mode_differences
        
        print(f"✅ Cross-asset analysis completed")
        return aggregated
    
    def _create_cross_asset_visualizations(self, aggregated: Dict, both_modes: bool = True):
        """Create and auto-save cross-asset visualization summaries"""
        print("📊 Creating cross-asset visualizations...")
        
        # 1. Common patterns heatmap
        self._plot_common_patterns_heatmap(aggregated['common_patterns'], both_modes)
        
        # 2. Mode comparison chart (if both modes)
        if both_modes and aggregated['mode_comparison']:
            self._plot_mode_comparison_chart(aggregated['mode_comparison'])
        
        # 3. Edge frequency network
        self._plot_edge_frequency_network(aggregated['common_patterns'])
    
    def _plot_common_patterns_heatmap(self, common_patterns: Dict, both_modes: bool):
        """Plot heatmap of common causal patterns across assets"""
        if not common_patterns:
            print("   No common patterns to visualize")
            return
        
        # Collect all unique edges across all algorithm-mode combinations
        all_unique_edges = set()
        for patterns in common_patterns.values():
            all_unique_edges.update(patterns.keys())
        
        if not all_unique_edges:
            print("   No common patterns found across assets")
            return
        
        # Create matrix: rows = edges, columns = algorithm-mode combinations
        all_unique_edges = sorted(list(all_unique_edges))
        algo_mode_keys = sorted(common_patterns.keys())
        
        matrix = np.zeros((len(all_unique_edges), len(algo_mode_keys)))
        
        for j, algo_mode_key in enumerate(algo_mode_keys):
            for i, edge in enumerate(all_unique_edges):
                if edge in common_patterns[algo_mode_key]:
                    matrix[i, j] = len(common_patterns[algo_mode_key][edge])
        
        # Plot heatmap
        plt.figure(figsize=(12, max(8, len(all_unique_edges) * 0.3)))
        
        im = plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        plt.yticks(range(len(all_unique_edges)), 
                   [edge[:40] + '...' if len(edge) > 40 else edge for edge in all_unique_edges])
        plt.xticks(range(len(algo_mode_keys)), algo_mode_keys, rotation=45, ha='right')
        
        plt.xlabel('Algorithm-Mode Combination')
        plt.ylabel('Causal Relationships')
        plt.title('Common Causal Patterns Across Assets\n(Color intensity = Number of assets showing pattern)')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Number of Assets')
        
        plt.tight_layout()
        
        # Auto-save
        save_path = self._get_save_path('cross_asset', 'common_patterns_heatmap.png')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Common patterns heatmap saved to {save_path}")
        
        plt.show()
    
    def _plot_mode_comparison_chart(self, mode_comparison: Dict):
        """Plot comparison between forced and free modes"""
        if not mode_comparison:
            return
        
        algorithms = list(mode_comparison.keys())
        
        # Prepare data for plotting
        forced_totals = [mode_comparison[algo]['forced_total'] for algo in algorithms]
        free_totals = [mode_comparison[algo]['free_total'] for algo in algorithms]
        overlap_ratios = [mode_comparison[algo]['overlap_ratio'] for algo in algorithms]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Total edges comparison
        x = np.arange(len(algorithms))
        width = 0.35
        
        ax1.bar(x - width/2, forced_totals, width, label='Forced Mode', color='#FF6B6B', alpha=0.8)
        ax1.bar(x + width/2, free_totals, width, label='Free Mode', color='#4ECDC4', alpha=0.8)
        
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Total Edges Found')
        ax1.set_title('Total Edges: Forced vs Free Mode')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Overlap ratios
        bars = ax2.bar(algorithms, overlap_ratios, color='#96CEB4', alpha=0.8)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Overlap Ratio')
        ax2.set_title('Mode Overlap Ratio\n(Higher = More Similar Results)')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, overlap_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Auto-save
        save_path = self._get_save_path('cross_asset', 'mode_comparison_chart.png')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Mode comparison chart saved to {save_path}")
        
        plt.show()
    
    def _plot_edge_frequency_network(self, common_patterns: Dict):
        """Plot network showing most frequent causal relationships"""
        # Aggregate all patterns across algorithms/modes
        edge_frequencies = {}
        
        for patterns in common_patterns.values():
            for edge, assets in patterns.items():
                if edge not in edge_frequencies:
                    edge_frequencies[edge] = 0
                edge_frequencies[edge] += len(assets)
        
        # Get top edges by frequency
        top_edges = sorted(edge_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if not top_edges:
            print("   No edges to plot in frequency network")
            return
        
        # Create network graph
        G = nx.DiGraph()
        
        for edge, frequency in top_edges:
            source, target = edge.split(' -> ')
            G.add_edge(source, target, weight=frequency)
        
        plt.figure(figsize=(16, 12))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Draw nodes
        node_sizes = [300 + G.degree(node) * 100 for node in G.nodes()]
        node_colors = [self._get_node_color_by_name(node) for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.8)
        
        # Draw edges with thickness proportional to frequency
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        edge_widths = [3 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                              edge_color='gray', arrowsize=20)
        
        # Draw labels
        labels = {node: node[:15] + '...' if len(node) > 15 else node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title('Most Frequent Causal Relationships Across Assets\n(Edge thickness = Frequency)', 
                 size=16, weight='bold', pad=20)
        plt.axis('off')
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='#FF6B6B', s=100, label='Macro Features', alpha=0.8),
            plt.scatter([], [], c='#4ECDC4', s=100, label='Technical Features', alpha=0.8),
            plt.scatter([], [], c='#45B7D1', s=100, label='Momentum Features', alpha=0.8),
            plt.scatter([], [], c='#96CEB4', s=100, label='Volatility Features', alpha=0.8),
            plt.scatter([], [], c='#FFEAA7', s=100, label='Volume Features', alpha=0.8),
            plt.scatter([], [], c='#DDA0DD', s=100, label='Cross-Asset Features', alpha=0.8),
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.tight_layout()
        
        # Auto-save
        save_path = self._get_save_path('cross_asset', 'edge_frequency_network.png')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Edge frequency network saved to {save_path}")
        
        plt.show()
    
    def _get_node_color_by_name(self, node_name: str) -> str:
        """Get color for a node based on feature category"""
        color_map = {
            'macro': '#FF6B6B',      # Red
            'technical': '#4ECDC4',   # Teal
            'momentum': '#45B7D1',    # Blue
            'volatility': '#96CEB4',  # Green
            'volume': '#FFEAA7',      # Yellow
            'cross_asset': '#DDA0DD',  # Plum
            'default': '#95A5A6'       # Gray
        }
        
        for cat, features in self.domain_knowledge['feature_categories'].items():
            if any(feat in node_name for feat in features):
                return color_map.get(cat, color_map['default'])
        
        return color_map['default']
    
    def save_results(self, save_dir: str = None):
        """Save all causal discovery results"""
        if save_dir is None:
            save_dir = self.data_dir
        
        results_path = os.path.join(save_dir, 'causal_results.pkl')
        
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary
        summary_path = os.path.join(save_dir, 'causal_summary.json')
        
        # Parse results by ticker and mode
        results_by_ticker = {}
        for result_key, result in self.results.items():
            if '_' in result_key:
                ticker, mode = result_key.rsplit('_', 1)
                if ticker not in results_by_ticker:
                    results_by_ticker[ticker] = {}
                results_by_ticker[ticker][mode] = result
        
        summary = {
            'assets_analyzed': list(results_by_ticker.keys()),
            'modes_analyzed': list(set(mode for ticker_data in results_by_ticker.values() for mode in ticker_data.keys())),
            'algorithms_used': list(set(
                algo for result in self.results.values() 
                for algo in result['algorithms'].keys()
            )),
            'analysis_date': datetime.now().isoformat(),
            'total_graphs': sum(len(result['algorithms']) for result in self.results.values()),
            'results_structure': {ticker: list(modes.keys()) for ticker, modes in results_by_ticker.items()},
            'visualization_directory': self.viz_dir if self.auto_save else None
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Results saved to {save_dir}")
        print(f"   Results: {results_path}")
        print(f"   Summary: {summary_path}")
        if self.auto_save:
            print(f"   Visualizations: {self.viz_dir}")
    
    def load_results(self, save_dir: str = None):
        """Load previously saved results"""
        if save_dir is None:
            save_dir = self.data_dir
        
        results_path = os.path.join(save_dir, 'causal_results.pkl')
        
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                self.results = pickle.load(f)
            print(f"✅ Loaded results for {len(self.results)} asset-mode combinations")
        else:
            print(f"❌ No saved results found at {results_path}")
    
    def print_file_structure(self):
        """Print the saved file structure - useful for Colab users"""
        if not self.auto_save:
            print("🚫 Auto-save is disabled. No files saved.")
            return
        
        print("📁 Saved File Structure:")
        print(f"📂 {self.data_dir}/")
        print("   📄 causal_results.pkl")
        print("   📄 causal_summary.json") 
        print("   📂 visualizations/")
        
        viz_subdirs = ['individual_graphs', 'algorithm_comparisons', 'mode_comparisons', 'summary_reports', 'cross_asset']
        
        for subdir in viz_subdirs:
            subdir_path = os.path.join(self.viz_dir, subdir)
            if os.path.exists(subdir_path):
                files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
                print(f"      📂 {subdir}/ ({len(files)} files)")
                for file in sorted(files)[:3]:  # Show first 3 files
                    print(f"         📊 {file}")
                if len(files) > 3:
                    print(f"         ... and {len(files) - 3} more files")
        
        print(f"\n🔍 In Colab, access files via the file browser on the left")
        print(f"📥 Or download the entire '{os.path.basename(self.data_dir)}' folder")


# Updated main function with auto-save demonstrations
def main():
    """
    Example usage of the FinancialCausalGraphBuilder with AUTO-SAVE
    Perfect for Google Colab!
    """
    print("🚀 Financial Causal Graph Builder Demo - AUTO-SAVE VERSION")
    print("=" * 60)
    print("🏷️  Perfect for Google Colab - All visualizations auto-saved!")
    print("=" * 60)
    
    # Initialize the builder with auto-save enabled (default)
    try:
        builder = FinancialCausalGraphBuilder(data_dir='data', auto_save=True)
        
        if builder.data_tensor is None:
            print("📊 No existing data found. Generating sample data...")
            
            # Create sample data using the collector
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']
            collector = CausalPortfolioDataCollector(
                tickers=tickers,
                start_date='2020-01-01',
                end_date='2023-01-01'
            )
            
            # Run the pipeline with synthetic data
            data_tensor, final_tickers, dates = collector.run_causal_pipeline(
                save_dir='data',
                use_synthetic=True
            )
            
            # Reload the builder with new data
            builder = FinancialCausalGraphBuilder(data_dir='data', auto_save=True)
        
        # Demonstrate causal discovery for a single asset
        print("\n" + "="*50)
        print("🎯 SINGLE ASSET CAUSAL DISCOVERY - BOTH MODES")
        print("="*50)
        
        # Choose first available ticker
        demo_ticker = builder.tickers[0]
        print(f"\n📈 Analyzing {demo_ticker}...")
        
        # Run causal discovery for both modes
        algorithms = ['PC', 'GES', 'ICALiNGAM']
        
        # Forced mode (includes log_return)
        print(f"\n🔴 Running FORCED mode for {demo_ticker}...")
        forced_result = builder.discover_causal_graph(
            ticker=demo_ticker, 
            algorithms=algorithms,
            max_features=12,
            force_log_return=True
        )
        
        # Free mode (pure feature selection)
        print(f"\n🔵 Running FREE mode for {demo_ticker}...")
        free_result = builder.discover_causal_graph(
            ticker=demo_ticker, 
            algorithms=algorithms,
            max_features=12,
            force_log_return=False
        )
        
        # Visualize results for both modes (AUTO-SAVED!)
        print(f"\n📊 Creating and auto-saving visualizations for {demo_ticker}...")
        
        for algorithm in algorithms[:2]:  # Limit to 2 algorithms for demo
            if algorithm in forced_result['algorithms']:
                print(f"   📊 {algorithm} (Forced mode) - AUTO-SAVING...")
                builder.visualize_causal_graph(
                    ticker=demo_ticker, 
                    algorithm=algorithm,
                    mode='forced',
                    figsize=(10, 8)
                )
            
            if algorithm in free_result['algorithms']:
                print(f"   📊 {algorithm} (Free mode) - AUTO-SAVING...")
                builder.visualize_causal_graph(
                    ticker=demo_ticker, 
                    algorithm=algorithm,
                    mode='free',
                    figsize=(10, 8)
                )
        
        # Compare algorithms within each mode (AUTO-SAVED!)
        print(f"\n📊 Creating algorithm comparisons for {demo_ticker}...")
        print("   🔴 Forced mode comparison - AUTO-SAVING...")
        builder.compare_algorithms(demo_ticker, mode='forced')
        
        print("   🔵 Free mode comparison - AUTO-SAVING...")
        builder.compare_algorithms(demo_ticker, mode='free')
        
        # Compare modes for the same algorithm (AUTO-SAVED!)
        print(f"\n📊 Creating mode comparison for {demo_ticker} - AUTO-SAVING...")
        builder.compare_modes(demo_ticker, algorithm='GES')
        
        # Cross-asset analysis (if multiple tickers available)
        if len(builder.tickers) > 1:
            print("\n" + "="*50)
            print("🌍 CROSS-ASSET CAUSALITY ANALYSIS - BOTH MODES")
            print("="*50)
            print("🎨 Creating cross-asset visualizations - AUTO-SAVING...")
            
            cross_results = builder.analyze_cross_asset_causality(
                algorithms=['GES', 'Granger'],
                max_features=8,
                run_both_modes=True
            )
            
            print("\n📋 Cross-Asset Analysis Results:")
            print("\n🔍 Common Causal Patterns Found:")
            for algo_mode, patterns in cross_results['common_patterns'].items():
                if patterns:  # Only show if patterns exist
                    print(f"\n{algo_mode}:")
                    for pattern, assets in list(patterns.items())[:3]:  # Show top 3
                        print(f"   {pattern} (found in {len(assets)} assets)")
            
            print("\n⚖️  Mode Comparison Analysis:")
            for algorithm, comparison in cross_results['mode_comparison'].items():
                print(f"\n{algorithm}:")
                print(f"   Forced only edges: {len(comparison['forced_only'])}")
                print(f"   Free only edges: {len(comparison['free_only'])}")
                print(f"   Common edges: {len(comparison['common'])}")
                print(f"   Overlap ratio: {comparison['overlap_ratio']:.2%}")
        
        # Save all results
        print(f"\n💾 Saving all results and metadata...")
        builder.save_results()
        
        # Show file structure
        print(f"\n📁 SAVED FILE STRUCTURE:")
        builder.print_file_structure()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"   📊 Analyzed: {len(set(k.split('_')[0] for k in builder.results.keys()))} assets")
        print(f"   🎯 Total analysis runs: {len(builder.results)}")
        print(f"   🎨 All visualizations auto-saved!")
        print(f"   📁 Access files in Colab file browser: {builder.viz_dir}")
        
        # Colab-specific instructions
        print(f"\n📱 GOOGLE COLAB INSTRUCTIONS:")
        print(f"   1️⃣ Click the folder icon on the left sidebar")
        print(f"   2️⃣ Navigate to: data/visualizations/")
        print(f"   3️⃣ Right-click any .png file → Download")
        print(f"   4️⃣ Or download entire 'data' folder as ZIP")
        
        return builder
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_causal_analysis_pipeline():
    """
    Create a complete causal analysis pipeline for financial data
    AUTO-SAVE VERSION for Google Colab
    """
    class CausalAnalysisPipeline:
        def __init__(self, tickers, start_date='2020-01-01', end_date='2024-01-01'):
            self.tickers = tickers
            self.start_date = start_date
            self.end_date = end_date
            self.collector = None
            self.builder = None
            
        def run_complete_pipeline(self, use_real_data=True, save_dir='data', run_both_modes=True):
            """Run the complete pipeline from data collection to causal discovery with AUTO-SAVE"""
            print("🚀 Running Complete Causal Analysis Pipeline - AUTO-SAVE VERSION")
            print("=" * 70)
            
            # Step 1: Data Collection
            print("\n📊 STEP 1: DATA COLLECTION")
            print("-" * 30)
            
            self.collector = CausalPortfolioDataCollector(
                tickers=self.tickers,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Custom features for better causal discovery
            causal_features = [
                # Core financial features
                'return', 'log_return', 'volatility', 'momentum_10',
                
                # Technical indicators
                'rsi', 'macd', 'bb_position', 'atr',
                
                # Market context
                'macro_sp500_return', 'macro_vix_level', 'macro_vix_change',
                'macro_treasury_10y_return', 'macro_gold_return',
                
                # Cross-asset relationships
                'cross_vix_regime', 'cross_beta_sp500'
            ]
            
            try:
                data_tensor, tickers, dates = self.collector.run_causal_pipeline(
                    feature_list=causal_features,
                    save_dir=save_dir,
                    use_synthetic=not use_real_data
                )
                print(f"✅ Data collection completed: {data_tensor.shape}")
                
            except Exception as e:
                print(f"❌ Data collection failed: {e}")
                print("🔄 Falling back to synthetic data...")
                data_tensor, tickers, dates = self.collector.run_causal_pipeline(
                    feature_list=causal_features,
                    save_dir=save_dir,
                    use_synthetic=True
                )
            
            # Step 2: Causal Discovery with AUTO-SAVE
            print("\n🔍 STEP 2: CAUSAL DISCOVERY + AUTO-SAVE")
            print("-" * 40)
            
            self.builder = FinancialCausalGraphBuilder(data_dir=save_dir, auto_save=True)
            
            # Discover causal graphs for all assets
            all_results = {}
            algorithms = ['PC', 'GES', 'ICALiNGAM']
            
            modes_to_run = [True, False] if run_both_modes else [True]
            
            for force_mode in modes_to_run:
                mode_name = "forced" if force_mode else "free"
                print(f"   Running {mode_name} mode...")
                
                try:
                    result = self.builder.discover_causal_graph(
                        ticker=ticker,
                        algorithms=algorithms,
                        max_features=10,
                        force_log_return=force_mode
                    )
                    result_key = f"{ticker}_{mode_name}"
                    all_results[result_key] = result
                    
                    # Auto-create individual visualizations
                    print(f"      Creating visualizations for {ticker} ({mode_name})...")
                    for algorithm in algorithms[:2]:  # Limit for demo
                        if algorithm in result['algorithms']:
                            self.builder.visualize_causal_graph(
                                ticker=ticker, 
                                algorithm=algorithm,
                                mode=mode_name,
                                figsize=(8, 6)
                            )
                except Exception as e:
                    print(f"⚠️  Failed to analyze {ticker} in {mode_name} mode: {e}")
                    continue
            
            # Step 3: Visualization and Analysis with AUTO-SAVE
            print("\n📊 STEP 3: COMPREHENSIVE VISUALIZATION + AUTO-SAVE")
            print("-" * 50)
            
            # Create summary visualizations
            self.create_summary_report(all_results, save_dir, run_both_modes)
            
            # Create algorithm and mode comparisons
            print("📊 Creating detailed comparisons...")
            analyzed_tickers = list(set(k.split('_')[0] for k in all_results.keys()))
            
            for ticker in analyzed_tickers[:2]:  # Limit for demo
                # Algorithm comparisons for each mode
                if run_both_modes:
                    for mode in ['forced', 'free']:
                        result_key = f"{ticker}_{mode}"
                        if result_key in all_results:
                            print(f"   📊 {ticker} - {mode} mode algorithm comparison...")
                            self.builder.compare_algorithms(ticker, mode=mode)
                    
                    # Mode comparison
                    print(f"   📊 {ticker} - mode comparison...")
                    self.builder.compare_modes(ticker, algorithm='GES')
                else:
                    self.builder.compare_algorithms(ticker, mode='forced')
            
            # Step 4: Cross-Asset Analysis with AUTO-SAVE
            if len(analyzed_tickers) > 1:
                print("\n🌍 STEP 4: CROSS-ASSET ANALYSIS + AUTO-SAVE")
                print("-" * 45)
                
                cross_results = self.builder.analyze_cross_asset_causality(
                    algorithms=['GES', 'Granger'],
                    max_features=8,
                    run_both_modes=run_both_modes
                )
            
            # Step 5: Save Everything
            print("\n💾 STEP 5: SAVING ALL RESULTS")
            print("-" * 30)
            
            self.builder.save_results(save_dir)
            
            # Show final file structure
            print(f"\n📁 FINAL FILE STRUCTURE:")
            self.builder.print_file_structure()
            
            print(f"\n🎉 Complete pipeline finished!")
            print(f"   📊 Assets analyzed: {len(analyzed_tickers)}")
            print(f"   🎯 Total analysis runs: {len(all_results)}")
            print(f"   🎨 All visualizations auto-saved!")
            print(f"   📁 Ready for download from: {save_dir}/")
            
            return all_results
        
        def create_summary_report(self, results, save_dir, both_modes=True):
            """Create a summary report of all causal discoveries - AUTO-SAVE VERSION"""
            print("📋 Creating comprehensive summary report...")
            
            # Aggregate statistics
            total_edges = {}
            algorithm_comparison = {}
            mode_comparison = {}
            
            for result_key, result in results.items():
                ticker, mode = result_key.rsplit('_', 1)
                
                for algorithm, algo_result in result['algorithms'].items():
                    algo_mode_key = f"{algorithm}_{mode}"
                    
                    if algo_mode_key not in total_edges:
                        total_edges[algo_mode_key] = []
                    if algorithm not in algorithm_comparison:
                        algorithm_comparison[algorithm] = {'forced': 0, 'free': 0}
                    
                    num_edges = algo_result.get('num_edges', 0)
                    total_edges[algo_mode_key].append(num_edges)
                    algorithm_comparison[algorithm][mode] += num_edges
            
            # Create comprehensive summary plot
            fig_height = 20 if both_modes else 15
            plt.figure(figsize=(20, fig_height))
            
            subplot_rows = 3 if both_modes else 2
            subplot_cols = 3
            
            # Subplot 1: Algorithm comparison by mode
            plt.subplot(subplot_rows, subplot_cols, 1)
            
            if both_modes:
                algorithms = list(algorithm_comparison.keys())
                forced_counts = [algorithm_comparison[alg]['forced'] for alg in algorithms]
                free_counts = [algorithm_comparison[alg]['free'] for alg in algorithms]
                
                x = np.arange(len(algorithms))
                width = 0.35
                
                plt.bar(x - width/2, forced_counts, width, label='Forced', 
                       color='#FF6B6B', alpha=0.8)
                plt.bar(x + width/2, free_counts, width, label='Free', 
                       color='#4ECDC4', alpha=0.8)
                
                plt.title('Total Edges by Algorithm and Mode', weight='bold')
                plt.xlabel('Algorithm')
                plt.ylabel('Total Edges')
                plt.xticks(x, algorithms, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                algorithms = list(algorithm_comparison.keys())
                edge_counts = [algorithm_comparison[alg]['forced'] for alg in algorithms]
                
                plt.bar(algorithms, edge_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                       alpha=0.8)
                plt.title('Total Edges Discovered by Algorithm', weight='bold')
                plt.ylabel('Total Edges')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
            
            # Subplot 2: Edge distribution
            plt.subplot(subplot_rows, subplot_cols, 2)
            for algo_mode, edges in total_edges.items():
                if edges:  # Only plot if we have data
                    plt.hist(edges, alpha=0.7, label=algo_mode, 
                            bins=max(3, min(10, len(edges))), density=True)
            plt.title('Edge Count Distribution', weight='bold')
            plt.xlabel('Edges per Asset')
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Asset comparison
            plt.subplot(subplot_rows, subplot_cols, 3)
            asset_names = list(set(k.split('_')[0] for k in results.keys()))
            asset_total_edges = []
            asset_mode_breakdown = {'forced': [], 'free': []} if both_modes else {'forced': []}
            
            for ticker in asset_names:
                total = 0
                mode_totals = {'forced': 0, 'free': 0}
                
                for result_key, result in results.items():
                    if result_key.startswith(ticker):
                        mode = result_key.split('_')[-1]
                        mode_total = sum(algo_result.get('num_edges', 0) 
                                       for algo_result in result['algorithms'].values())
                        total += mode_total
                        if mode in mode_totals:
                            mode_totals[mode] = mode_total
                
                asset_total_edges.append(total)
                for mode in asset_mode_breakdown:
                    asset_mode_breakdown[mode].append(mode_totals[mode])
            
            if both_modes:
                x = np.arange(len(asset_names))
                width = 0.35
                plt.bar(x - width/2, asset_mode_breakdown['forced'], width, 
                       label='Forced', color='#FF6B6B', alpha=0.8)
                plt.bar(x + width/2, asset_mode_breakdown['free'], width, 
                       label='Free', color='#4ECDC4', alpha=0.8)
                plt.xticks(x, asset_names, rotation=45)
                plt.legend()
            else:
                plt.bar(asset_names, asset_total_edges, color='#96CEB4', alpha=0.8)
                plt.xticks(rotation=45)
            
            plt.title('Total Edges by Asset', weight='bold')
            plt.xlabel('Asset')
            plt.ylabel('Total Edges')
            plt.grid(True, alpha=0.3)
            
            # Additional subplots for detailed analysis
            if both_modes:
                # Subplot 4: Mode overlap analysis
                plt.subplot(subplot_rows, subplot_cols, 4)
                
                overlap_stats = []
                asset_labels = []
                
                for ticker in asset_names:
                    forced_key = f"{ticker}_forced"
                    free_key = f"{ticker}_free"
                    
                    if forced_key in results and free_key in results:
                        forced_total = sum(r.get('num_edges', 0) for r in results[forced_key]['algorithms'].values())
                        free_total = sum(r.get('num_edges', 0) for r in results[free_key]['algorithms'].values())
                        
                        if forced_total > 0 or free_total > 0:
                            overlap_ratio = min(forced_total, free_total) / max(1, max(forced_total, free_total))
                            overlap_stats.append(overlap_ratio)
                            asset_labels.append(ticker)
                
                if overlap_stats:
                    bars = plt.bar(asset_labels, overlap_stats, color='#DDA0DD', alpha=0.8)
                    plt.title('Mode Overlap Ratio by Asset', weight='bold')
                    plt.ylabel('Overlap Ratio')
                    plt.xticks(rotation=45)
                    plt.ylim(0, 1)
                    plt.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, ratio in zip(bars, overlap_stats):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
                else:
                    plt.text(0.5, 0.5, 'Mode Overlap\nAnalysis\n(Insufficient Data)', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
                    plt.title('Mode Overlap Analysis', weight='bold')
                    plt.axis('off')
                
                # Subplot 5: Algorithm performance comparison
                plt.subplot(subplot_rows, subplot_cols, 5)
                
                algorithm_avg_edges = {}
                for algo in algorithm_comparison:
                    forced_avg = algorithm_comparison[algo]['forced'] / max(1, len([k for k in results.keys() if 'forced' in k]))
                    free_avg = algorithm_comparison[algo]['free'] / max(1, len([k for k in results.keys() if 'free' in k]))
                    algorithm_avg_edges[algo] = {'forced': forced_avg, 'free': free_avg}
                
                algorithms = list(algorithm_avg_edges.keys())
                forced_avgs = [algorithm_avg_edges[alg]['forced'] for alg in algorithms]
                free_avgs = [algorithm_avg_edges[alg]['free'] for alg in algorithms]
                
                x = np.arange(len(algorithms))
                width = 0.35
                
                plt.bar(x - width/2, forced_avgs, width, label='Forced', 
                       color='#FF6B6B', alpha=0.8)
                plt.bar(x + width/2, free_avgs, width, label='Free', 
                       color='#4ECDC4', alpha=0.8)
                
                plt.title('Average Edges per Asset by Algorithm', weight='bold')
                plt.xlabel('Algorithm')
                plt.ylabel('Average Edges')
                plt.xticks(x, algorithms, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Subplot 6: Feature category analysis placeholder
                plt.subplot(subplot_rows, subplot_cols, 6)
                plt.text(0.5, 0.5, 'Feature Category\nInfluence Analysis\n(Advanced Analytics)', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                plt.title('Feature Category Analysis', weight='bold')
                plt.axis('off')
                
            else:
                # Single mode additional analysis
                plt.subplot(subplot_rows, subplot_cols, 4)
                
                # Algorithm efficiency comparison
                algorithm_efficiency = {}
                for algo in algorithm_comparison:
                    total_edges = algorithm_comparison[algo]['forced']
                    num_assets = len(set(k.split('_')[0] for k in results.keys()))
                    efficiency = total_edges / max(1, num_assets)
                    algorithm_efficiency[algo] = efficiency
                
                algorithms = list(algorithm_efficiency.keys())
                efficiencies = list(algorithm_efficiency.values())
                
                bars = plt.bar(algorithms, efficiencies, 
                              color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
                plt.title('Algorithm Efficiency\n(Avg Edges per Asset)', weight='bold')
                plt.ylabel('Average Edges')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, eff in zip(bars, efficiencies):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
            
            mode_text = "Dual Mode" if both_modes else "Single Mode"
            plt.suptitle(f'Comprehensive Causal Discovery Analysis - {mode_text}', 
                        size=20, weight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Auto-save comprehensive summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, 'visualizations', 'summary_reports', 
                                   f'comprehensive_summary_{mode_text.lower().replace(" ", "_")}_{timestamp}.png')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comprehensive summary report saved to {save_path}")
            
            plt.show()
    
    return CausalAnalysisPipeline


# Colab-specific helper functions
def download_results_colab(data_dir='data'):
    """
    Helper function for Google Colab users to create a downloadable ZIP
    """
    try:
        import zipfile
        import os
        
        zip_filename = f"causal_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_path = os.path.relpath(file_path, data_dir)
                    zipf.write(file_path, arc_path)
        
        print(f"📦 Created downloadable ZIP: {zip_filename}")
        print(f"📥 Right-click the file in Colab's file browser to download")
        
        return zip_filename
    
    except ImportError:
        print("❌ ZIP creation failed. Download files individually from the file browser.")
        return None


def setup_colab_environment():
    """
    Setup function for Google Colab users
    """
    print("🚀 Setting up Google Colab environment for Causal Analysis...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    
    # Install required packages if needed (uncomment if running in fresh Colab)
    """
    !pip install causal-learn
    !pip install networkx
    !pip install statsmodels
    """
    
    print("✅ Colab environment ready!")
    print("📁 Data directory created: ./data/")
    print("🎨 Visualizations will be auto-saved to: ./data/visualizations/")
    print("\n🎯 Run main() to start the causal analysis demo!")


if __name__ == "__main__":
    # For Google Colab users
    print("🏷️  GOOGLE COLAB VERSION - AUTO-SAVE ENABLED")
    print("="*50)
    
    # Setup environment
    setup_colab_environment()
    
    # Run the main demo with auto-save
    builder = main()
    
    # Create downloadable ZIP for easy access
    if builder is not None:
        print(f"\n📦 Creating downloadable package...")
        zip_file = download_results_colab('data')
    
    # Alternative: Run complete pipeline
    if False:  # Set to True to run complete pipeline
        # Define test portfolio
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM', 'GLD', 'SPY']
        
        # Create and run pipeline
        pipeline_class = create_causal_analysis_pipeline()
        pipeline = pipeline_class(
            tickers=test_tickers,
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        
        results = pipeline.run_complete_pipeline(
            use_real_data=False,  # Set to True to use real data
            save_dir='data',
            run_both_modes=True   # Run both forced and free modes
        )
        
        # Create downloadable package
        print(f"\n📦 Creating final downloadable package...")
        zip_file = download_results_colab('data')