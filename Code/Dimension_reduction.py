'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-12-01 16:51:01
LastEditors: RemoteScy 98257102+Schuyn@users.noreply.github.com
LastEditTime: 2025-12-02 17:22:43
FilePath: /Unsupervised-Machine-Learning-Final-Project/Code/Dimension_reduction.py
Description: 
    Dimensionality Reduction Module for NBA Draft Analysis

    This module handles:
    - PCA analysis and visualization
    - Explained variance computation
    - Component interpretation
    - Saving reduced-dimension data
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pickle
import os
from typing import Tuple, List, Optional, Dict
import umap

class Analyzer:
    """
    PCA analysis and visualization for NBA draft data
    """
    def __init__(self, output_dir: str = 'Data/processed', 
                 figure_dir: str = 'Latex/Figure'):
        """
        Initialize PCA analyzer
        
        Args:
            output_dir: Directory to save processed data
            figure_dir: Directory to save figures
        """
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        self.pca_full = None
        self.explained_variance = None
        self.cumulative_variance = None
        self.feature_names = None
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)
    
    def _save_metric_points(self, coords: np.ndarray, metric: str, metric_values: np.ndarray, 
                        quartile_colors: List[int], reducer_name: str, raw_data: pd.DataFrame):
        """
        Save 2D coordinates with metric values and quartile assignment, along with original data.
        
        Args:
            coords: 2D coordinates from dimensionality reduction
            metric: Name of the metric
            metric_values: Metric values for each point
            quartile_colors: Quartile assignment (0-3) for each point
            reducer_name: 'pca' or 'umap'
            raw_data: Complete original DataFrame with all player info
        """
        out_dir = os.path.join(self.output_dir, reducer_name, metric)
        os.makedirs(out_dir, exist_ok=True)
        
        df_out = raw_data.reset_index(drop=True).copy()
        df_out[f'{reducer_name}1'] = coords[:, 0]
        df_out[f'{reducer_name}2'] = coords[:, 1]
        df_out['metric_value'] = metric_values
        df_out['quartile'] = quartile_colors
        
        for q in range(4):
            df_out[df_out['quartile'] == q].to_csv(
                os.path.join(out_dir, f'quartile_{q+1}.csv'),
                index=False
            )
    
    def fit(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit PCA on training data with all components
        
        Args:
            X_train: Training feature matrix (already scaled)
            feature_names: List of feature names for interpretation
        """
        self.feature_names = feature_names
        
        # Fit PCA with all components
        self.pca_full = PCA()
        self.pca_full.fit(X_train)
        
        # Store variance information
        self.explained_variance = self.pca_full.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance)
        
        print(f"Total components: {len(self.explained_variance)}")
        
        return self
    
    def print_variance_stats(self, thresholds: List[float] = [0.80, 0.85, 0.90, 0.95]):
        """
        Print explained variance statistics
        
        Args:
            thresholds: List of variance thresholds to check
        """
        for threshold in thresholds:
            n_components = np.argmax(self.cumulative_variance >= threshold) + 1
            actual_var = self.cumulative_variance[n_components - 1]
            print(f"Components for {threshold*100:.0f}% variance: {n_components} "
                  f"(actual: {actual_var*100:.2f}%)")
        
        print("\nFirst 10 components:")
        for i in range(min(10, len(self.explained_variance))):
            print(f"  PC{i+1}: {self.explained_variance[i]*100:.2f}% "
                  f"(cumulative: {self.cumulative_variance[i]*100:.2f}%)")
    
    def plot_explained_variance(self, save_name: str = 'pca_explained_variance'):
        """
        Create explained variance plots
        
        Args:
            save_name: Base name for saved figures
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Cumulative Explained Variance
        ax1 = axes[0]
        n_components = len(self.explained_variance)
        ax1.plot(range(1, n_components + 1), self.cumulative_variance * 100, 
                marker='o', linewidth=2, markersize=4, color='steelblue')
        
        # Add threshold lines
        thresholds = [80, 85, 90, 95]
        colors = ['green', 'orange', 'red', 'purple']
        for threshold, color in zip(thresholds, colors):
            n_comp = np.argmax(self.cumulative_variance >= threshold/100) + 1
            ax1.axhline(y=threshold, color=color, linestyle='--', 
                    linewidth=1.5, alpha=0.7,
                    label=f'{threshold}% ({n_comp} comp.)')
            ax1.axvline(x=n_comp, color=color, linestyle='--', 
                    linewidth=1.5, alpha=0.7)
        
        ax1.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold')
        ax1.set_title('PCA Cumulative Explained Variance', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, n_components + 1)
        ax1.set_ylim(0, 105)
        
        # Plot 2: First 10 Components Bar Chart
        ax2 = axes[1]
        n_bars = min(10, len(self.explained_variance))
        bars = ax2.bar(range(1, n_bars + 1), self.explained_variance[:n_bars] * 100, 
                    color='coral', edgecolor='black', alpha=0.8)
        
        # Add value labels
        for bar, var in zip(bars, self.explained_variance[:n_bars]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{var*100:.1f}%', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Top 10 Principal Components', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(1, n_bars + 1))
        ax2.set_xticklabels([f'PC{i}' for i in range(1, n_bars + 1)])
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, max(self.explained_variance[:n_bars]) * 100 * 1.15)
        
        plt.tight_layout()
        
        # Save only PNG
        png_path = os.path.join(self.figure_dir, f'{save_name}.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: {png_path}")
    
    def plot_scree(self, n_components: int = 20, save_name: str = 'pca_scree_plot'):
        """
        Create scree plot
        
        Args:
            n_components: Number of components to show
            save_name: Base name for saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_plot = min(n_components, len(self.explained_variance))
        ax.plot(range(1, n_plot + 1), 
                self.explained_variance[:n_plot] * 100,
                marker='o', markersize=8, linewidth=2, color='darkblue')
        
        ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
        ax.set_title('Scree Plot', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, n_plot + 1))
        ax.grid(alpha=0.3)
        
        # Highlight elbow
        elbow_threshold = 5.0
        elbow_point = np.argmax(self.explained_variance * 100 < elbow_threshold)
        if elbow_point > 0:
            ax.axvline(x=elbow_point, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7,
                      label=f'Elbow at PC{elbow_point}')
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save
        png_path = os.path.join(self.figure_dir, f'{save_name}.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: {png_path}")
    
    def print_component_loadings(self, n_components: int = 5, 
                                 n_features: int = 10):
        """
        Print feature loadings for top components
        
        Args:
            n_components: Number of components to analyze
            n_features: Number of top features to show per component
        """
        if self.feature_names is None:
            print("Feature names not available")
            return
        
        for i in range(min(n_components, len(self.explained_variance))):
            print(f"\n{'='*80}")
            print(f"PC{i+1} (explains {self.explained_variance[i]*100:.2f}% variance)")
            print(f"{'='*80}")
            
            loadings = self.pca_full.components_[i]
            abs_loadings = np.abs(loadings)
            top_indices = np.argsort(abs_loadings)[::-1][:n_features]
            
            print(f"\nTop {n_features} features:")
            for rank, idx in enumerate(top_indices, 1):
                feature_name = self.feature_names[idx]
                loading = loadings[idx]
                sign = '+' if loading > 0 else '-'
                print(f"  {rank:2d}. {feature_name:30s} {sign} {abs(loading):.4f}")
    
    def transform_and_save(self, X_train: np.ndarray, X_val: np.ndarray,
                          configs: Optional[Dict[str, int]] = None):
        """
        Transform data to different dimensions and save
        
        Args:
            X_train: Training data
            X_val: Validation data
            configs: Dict of {name: n_components}
        """
        if configs is None:
            # Default configurations
            n_90 = np.argmax(self.cumulative_variance >= 0.90) + 1
            configs = {
                'pca_2d': 2,
                'pca_3d': 3,
                'pca_10d': 10,
                'pca_90pct': n_90
            }
        
        for config_name, n_comp in configs.items():
            # Fit PCA with n_comp components
            pca_model = PCA(n_components=n_comp)
            X_train_pca = pca_model.fit_transform(X_train)
            X_val_pca = pca_model.transform(X_val)
            
            # Save transformed data
            train_path = os.path.join(self.output_dir, f'X_train_{config_name}.npy')
            val_path = os.path.join(self.output_dir, f'X_val_{config_name}.npy')
            np.save(train_path, X_train_pca)
            np.save(val_path, X_val_pca)
            
            # Save PCA model
            model_path = os.path.join(self.output_dir, f'{config_name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(pca_model, f)
            
            var_explained = pca_model.explained_variance_ratio_.sum() * 100
            print(f"\n{config_name}:")
            print(f"  Components: {n_comp}")
            print(f"  Variance: {var_explained:.2f}%")
            print(f"  Train: {X_train_pca.shape}")
            print(f"  Val: {X_val_pca.shape}")
            print(f"  Saved: {train_path}")
    
    def plot_2d_colored_by_metrics(self, X_train: np.ndarray, 
                                   raw_data: pd.DataFrame, metrics: List[str],
                                   metric_names: Optional[Dict[str, str]] = None,
                                   save_combined: str = 'pca_2d_advanced_metrics_quartiles',
                                   save_individual: bool = False):
        """
        Plot 2D PCA colored by advanced metrics quartiles

        Args:
            X_train: Training feature matrix to be reduced to 2D
            raw_data: Complete DataFrame with all original columns (player, year, team, 
                    college, statistics, etc.) to be saved alongside reduction results
            metrics: List of metric column names to visualize (must exist in raw_data)
            metric_names: Optional dict mapping column names to display names
            save_combined: Name for combined 2x2 figure
            save_individual: Whether to save individual figures
        """
        if metric_names is None:
            metric_names = {m: m.replace('_', ' ').title() for m in metrics}
        
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X_train)
        
        # Create 2x2 combined plot
        n_metrics = len(metrics)
        n_rows = 2
        n_cols = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 16))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Get metric values
            metric_values = raw_data[metric].values
            
            # Calculate quartiles
            q1, q2, q3 = np.percentile(metric_values, [25, 50, 75])
            
            # Assign quartile colors
            quartile_colors = []
            for val in metric_values:
                if val <= q1:
                    quartile_colors.append(0)
                elif val <= q2:
                    quartile_colors.append(1)
                elif val <= q3:
                    quartile_colors.append(2)
                else:
                    quartile_colors.append(3)
            
            self._save_metric_points(
                coords=X_2d,
                metric=metric,
                metric_values=metric_values,
                quartile_colors=quartile_colors,
                reducer_name='pca',
                raw_data=raw_data
            )

            # Scatter plot
            scatter = ax.scatter(
                X_2d[:, 0], 
                X_2d[:, 1],
                c=quartile_colors,
                cmap='RdYlGn',
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidth=0.3
            )
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2, 3])
            cbar.set_ticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
            cbar.set_label('Quartile', rotation=270, labelpad=20, fontweight='bold')
            
            # Labels and title
            ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
            ax.set_title(f'2D PCA colored by {metric_names[metric]}', 
                        fontsize=13, fontweight='bold', pad=10)
            ax.grid(alpha=0.3, linestyle='--')
            
            # Quartile threshold info
            info_text = f'Q1: {q1:.2f}\nQ2: {q2:.2f}\nQ3: {q3:.2f}'
            ax.text(0.02, 0.98, info_text, 
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save combined figure
        combined_path = os.path.join(self.figure_dir, f'{save_combined}.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nSaved combined figure: {combined_path}")
    
    def plot_2d_with_player_highlights(self, X_train: np.ndarray,
                                    display_features: pd.DataFrame,
                                    player_names: List[str],
                                    save_name: str = 'pca_2d_player_highlights'):
        """
        Plot 2D PCA with specific players highlighted and labeled
        
        Args:
            X_train: Original training data (will be transformed to 2D)
            display_features: DataFrame with player names and metadata
            player_names: List of player names to highlight
            save_name: Name for saved figure
        """
        # Transform to 2D
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X_train)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot all players in light gray
        ax.scatter(X_2d[:, 0], X_2d[:, 1],
                c='lightgray', alpha=0.3, s=20,
                edgecolors='none', label='Other players')
        
        # Highlight and label specific players
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        found_players = []
        
        for idx, player_name in enumerate(player_names):
            color = colors[idx % len(colors)]
            
            # Find player in display_features
            # Try exact match first
            mask = display_features['player'].str.contains(player_name, case=False, na=False)
            
            if mask.any():
                player_indices = mask[mask].index.tolist()
                
                for player_idx in player_indices:
                    # Get coordinates
                    x, y = X_2d[player_idx, 0], X_2d[player_idx, 1]
                    
                    # Plot highlighted point
                    ax.scatter(x, y, c=color, s=200, 
                            edgecolors='black', linewidth=2,
                            marker='*', zorder=10,
                            label=display_features.loc[player_idx, 'player'])
                    
                    # Add label with arrow
                    ax.annotate(display_features.loc[player_idx, 'player'],
                            xy=(x, y),
                            xytext=(15, 15),
                            textcoords='offset points',
                            fontsize=11,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', 
                                    facecolor=color, 
                                    alpha=0.7,
                                    edgecolor='black',
                                    linewidth=1.5),
                            arrowprops=dict(arrowstyle='->', 
                                            connectionstyle='arc3,rad=0.3',
                                            color='black',
                                            linewidth=2))
                    
                    found_players.append(display_features.loc[player_idx, 'player'])
                    print(f"Found: {display_features.loc[player_idx, 'player']} at index {player_idx}")
            else:
                print(f"Warning: Player '{player_name}' not found in dataset")
        
        # Labels and title
        ax.set_xlabel('Principal Component 1', fontsize=14, fontweight='bold')
        ax.set_ylabel('Principal Component 2', fontsize=14, fontweight='bold')
        ax.set_title('2D PCA: NBA Draft Players with Star Highlights', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.figure_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nSaved: {save_path}")
        print(f"Highlighted {len(found_players)} players: {', '.join(found_players)}")

    def plot_umap_colored_by_metrics(self, X_train: np.ndarray,
                                     raw_data: pd.DataFrame,
                                     metrics: List[str],
                                     metric_names: Optional[Dict[str, str]] = None,
                                     n_neighbors: int = 15,
                                     min_dist: float = 0.1,
                                     save_name: str = 'umap_2d_advanced_metrics'):
        """
        UMAP 2D reduction with advanced metrics quartile coloring
        
        Args:
            X_train: Training data
            display_features: DataFrame with metrics columns
            metrics: List of metric column names
            metric_names: Optional dict mapping column names to display names
            n_neighbors: UMAP n_neighbors parameter (default: 15)
            min_dist: UMAP min_dist parameter (default: 0.1)
            save_name: Name for saved figure
        """
        if metric_names is None:
            metric_names = {m: m.replace('_', ' ').title() for m in metrics}
        
        print("\n" + "="*80)
        print("UMAP 2D - COLORED BY ADVANCED METRICS")
        print("="*80)
        print(f"Input shape: {X_train.shape}")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
        
        # Fit UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=False
        )
        X_umap = reducer.fit_transform(X_train)
        
        print(f"UMAP embedding shape: {X_umap.shape}")
        
        # Create 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Get metric values
            metric_values = raw_data[metric].values
            
            # Calculate quartiles
            q1, q2, q3 = np.percentile(metric_values, [25, 50, 75])
            
            # Assign quartile colors
            quartile_colors = []
            for val in metric_values:
                if val <= q1:
                    quartile_colors.append(0)
                elif val <= q2:
                    quartile_colors.append(1)
                elif val <= q3:
                    quartile_colors.append(2)
                else:
                    quartile_colors.append(3)
            
            self._save_metric_points(
                coords=X_umap,
                metric=metric,
                metric_values=metric_values,
                quartile_colors=quartile_colors,
                reducer_name='umap',
                raw_data=raw_data
            )
            
            # Scatter plot
            scatter = ax.scatter(
                X_umap[:, 0], 
                X_umap[:, 1],
                c=quartile_colors,
                cmap='RdYlGn',
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidth=0.3
            )
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2, 3])
            cbar.set_ticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
            cbar.set_label('Quartile', rotation=270, labelpad=20, fontweight='bold')
            
            # Labels and title
            ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
            ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
            ax.set_title(f'UMAP colored by {metric_names[metric]}', 
                         fontsize=13, fontweight='bold', pad=10)
            ax.grid(alpha=0.3, linestyle='--')
            
            # Quartile threshold info
            info_text = f'Q1: {q1:.2f}\nQ2: {q2:.2f}\nQ3: {q3:.2f}'
            ax.text(0.02, 0.98, info_text, 
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.figure_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nSaved: {save_path}")
        
        return X_umap
    
    def plot_umap_with_player_highlights(self, X_train: np.ndarray,
                                          display_features: pd.DataFrame,
                                          player_names: List[str],
                                          n_neighbors: int = 15,
                                          min_dist: float = 0.1,
                                          save_name: str = 'umap_2d_star_players'):
        """
        UMAP 2D reduction with specific players highlighted
        
        Args:
            X_train: Training data
            display_features: DataFrame with player column
            player_names: List of player names to highlight
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            save_name: Name for saved figure
        """
        
        print("\n" + "="*80)
        print("UMAP WITH PLAYER HIGHLIGHTS")
        print("="*80)
        print(f"Input shape: {X_train.shape}")
        
        # Fit UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=False
        )
        X_umap = reducer.fit_transform(X_train)
        
        print(f"UMAP embedding shape: {X_umap.shape}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot all players in light gray
        ax.scatter(X_umap[:, 0], X_umap[:, 1],
                   c='lightgray', alpha=0.3, s=20,
                   edgecolors='none', label='Other players')
        
        # Highlight and label specific players
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        found_players = []
        
        for idx, player_name in enumerate(player_names):
            color = colors[idx % len(colors)]
            
            # Find player in display_features
            mask = display_features['player'].str.contains(player_name, case=False, na=False)
            
            if mask.any():
                player_indices = mask[mask].index.tolist()
                
                for player_idx in player_indices:
                    # Get coordinates
                    x, y = X_umap[player_idx, 0], X_umap[player_idx, 1]
                    
                    # Plot highlighted point
                    ax.scatter(x, y, c=color, s=200, 
                              edgecolors='black', linewidth=2,
                              marker='*', zorder=10,
                              label=display_features.loc[player_idx, 'player'])
                    
                    # Add label with arrow
                    ax.annotate(display_features.loc[player_idx, 'player'],
                               xy=(x, y),
                               xytext=(15, 15),
                               textcoords='offset points',
                               fontsize=11,
                               fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', 
                                       facecolor=color, 
                                       alpha=0.7,
                                       edgecolor='black',
                                       linewidth=1.5),
                               arrowprops=dict(arrowstyle='->', 
                                             connectionstyle='arc3,rad=0.3',
                                             color='black',
                                             linewidth=2))
                    
                    found_players.append(display_features.loc[player_idx, 'player'])
                    print(f"Found: {display_features.loc[player_idx, 'player']} at index {player_idx}")
            else:
                print(f"Warning: Player '{player_name}' not found in dataset")
        
        # Labels and title
        ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
        ax.set_title('UMAP: NBA Draft Players with Star Highlights', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.figure_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nSaved: {save_path}")
        print(f"Highlighted {len(found_players)} players: {', '.join(found_players)}")
        
        return X_umap
    
    def run_full_analysis(self, X_train: np.ndarray, X_val: np.ndarray,
                         feature_names: Optional[List[str]] = None):
        """
        Run complete PCA analysis pipeline
        
        Args:
            X_train: Training data
            X_val: Validation data
            feature_names: List of feature names
        """
        # Fit PCA
        self.fit(X_train, feature_names)
        
        # Print statistics
        self.print_variance_stats()
        
        # Create plots
        self.plot_explained_variance()
        self.plot_scree()
        
        # Print loadings
        if feature_names is not None:
            self.print_component_loadings()
        
        # Transform and save
        # self.transform_and_save(X_train, X_val)