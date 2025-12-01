'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-12-01 16:51:01
LastEditors: Schuyn 98257102+Schuyn@users.noreply.github.com
LastEditTime: 2025-12-01 16:56:53
FilePath: /Unsupervised-Machine-Learning-Final-Project/Code/PCA.py
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


class PCAAnalyzer:
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
        
        # Save
        png_path = os.path.join(self.figure_dir, f'{save_name}.png')
        pdf_path = os.path.join(self.figure_dir, f'{save_name}.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    
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
        pdf_path = os.path.join(self.figure_dir, f'{save_name}.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    
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
        self.transform_and_save(X_train, X_val)