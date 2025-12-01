'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-12-01 14:57:33
LastEditors: Schuyn 98257102+Schuyn@users.noreply.github.com
LastEditTime: 2025-12-01 16:29:38
FilePath: /Unsupervised-Machine-Learning-Final-Project/Code/Feature_engineering.py
Description: 
    This module handles:
    - Categorical encoding with on/off switches
    - Consistent column naming
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Optional
import pickle
import os


class NBAFeatureEngineer:
    """
    Feature engineering with flexible configuration for NBA draft data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with configuration
        
        Args:
            config: Configuration dictionary with feature switches
        """
        # Default configuration
        self.default_config = {
            'use_year': True,
            'use_team': True,
            'use_college': True,
            'college_top_n': 20,
            'scale_features': True,
            'include_draft_pick': True
        }
        
        self.config = config if config is not None else self.default_config
        
        # Storage for encoders and scalers
        self.encoders = {}
        self.scaler = None
        self.feature_names = []
    
    def _prepare_college(self, college_series: pd.Series, fit: bool) -> pd.Series:
        """
        Group colleges into top N + Other
        
        Args:
            college_series: Series of college names
            fit: Whether to identify top colleges
            
        Returns:
            Grouped college series
        """
        if fit:
            # Identify top N colleges
            college_counts = college_series.value_counts()
            top_n = self.config.get('college_top_n', 20)
            self.encoders['top_colleges'] = college_counts.head(top_n).index.tolist()
            print(f"  Identified {len(self.encoders['top_colleges'])} top colleges")
        
        # Apply grouping
        grouped = college_series.apply(
            lambda x: x if x in self.encoders['top_colleges'] else 'Other'
        )
        return grouped
    
    def create_features(self, df: pd.DataFrame, 
                       numeric_cols: List[str],
                       fit: bool = True) -> np.ndarray:
        """
        Create complete feature matrix
        
        Args:
            df: Input DataFrame
            numeric_cols: List of numeric column names
            fit: Whether to fit encoders
            
        Returns:
            Feature matrix as numpy array
        """
        if fit:
            self.feature_names = []
        
        feature_arrays = []
        
        # 1. Numeric features
        if numeric_cols:
            numeric_features = df[numeric_cols].values
            feature_arrays.append(numeric_features)
            if fit:
                self.feature_names.extend(numeric_cols)
            print(f"Added {len(numeric_cols)} numeric features")
        
        # 2. Draft pick
        if self.config.get('include_draft_pick', True) and 'overall_pick' in df.columns:
            draft_pick = df['overall_pick'].values.reshape(-1, 1)
            feature_arrays.append(draft_pick)
            if fit:
                self.feature_names.append('overall_pick')
            print("Added draft pick")
        
        # 3. Year (normalized to 0-1)
        if self.config.get('use_year', True) and 'year' in df.columns:
            if fit:
                self.encoders['year_min'] = df['year'].min()
                self.encoders['year_max'] = df['year'].max()
            
            year_normalized = (df['year'] - self.encoders['year_min']) / \
                            (self.encoders['year_max'] - self.encoders['year_min'])
            feature_arrays.append(year_normalized.values.reshape(-1, 1))
            
            if fit:
                self.feature_names.append('year_normalized')
            print("Added year (normalized)")
        
        # 4. Team (one-hot)
        if self.config.get('use_team', True) and 'team' in df.columns:
            team_df = pd.DataFrame({'team': df['team']})
            
            if fit:
                self.encoders['team'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                team_encoded = self.encoders['team'].fit_transform(team_df)
                team_names = self.encoders['team'].categories_[0]
                self.feature_names.extend([f'team_{t}' for t in team_names])
                print(f"Added {len(team_names)} team features (one-hot)")
            else:
                team_encoded = self.encoders['team'].transform(team_df)
            
            feature_arrays.append(team_encoded)
        
        # 5. College (grouped + one-hot)
        if self.config.get('use_college', True) and 'college' in df.columns:
            # Prepare college data
            college_grouped = self._prepare_college(df['college'], fit=fit)
            college_df = pd.DataFrame({'college': college_grouped})
            
            if fit:
                self.encoders['college'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                college_encoded = self.encoders['college'].fit_transform(college_df)
                college_names = self.encoders['college'].categories_[0]
                self.feature_names.extend([f'college_{c}' for c in college_names])
                print(f"Added {len(college_names)} college features (grouped + one-hot)")
            else:
                college_encoded = self.encoders['college'].transform(college_df)
            
            feature_arrays.append(college_encoded)
        
        # Concatenate all features
        X = np.hstack(feature_arrays)
        
        print(f"\nFinal feature matrix shape: {X.shape}")
        print(f"Total features: {len(self.feature_names)}")
        
        return X
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler
        
        Args:
            X: Feature matrix
            fit: Whether to fit scaler
            
        Returns:
            Scaled feature matrix
        """
        if not self.config.get('scale_features', True):
            return X
        
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            print("\nFeatures scaled using StandardScaler")
        else:
            X_scaled = self.scaler.transform(X)
            print("\nFeatures transformed using fitted scaler")
        
        return X_scaled
    
    def fit_transform(self, df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
        """
        Fit encoders and transform training data
        
        Args:
            df: Training DataFrame
            numeric_cols: List of numeric columns
            
        Returns:
            Transformed and scaled feature matrix
        """
        X = self.create_features(df, numeric_cols, fit=True)
        X_scaled = self.scale_features(X, fit=True)
        return X_scaled
    
    def transform(self, df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
        """
        Transform test/validation data using fitted encoders
        
        Args:
            df: Test DataFrame
            numeric_cols: List of numeric columns
            
        Returns:
            Transformed and scaled feature matrix
        """
        X = self.create_features(df, numeric_cols, fit=False)
        X_scaled = self.scale_features(X, fit=False)
        return X_scaled
    
    def save_artifacts(self, output_dir: str = 'Data/processed'):
        """Save encoders, scaler, and feature names"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save encoders
        with open(os.path.join(output_dir, 'encoders.pkl'), 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Save scaler
        if self.scaler is not None:
            with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save feature names
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            for fname in self.feature_names:
                f.write(f"{fname}\n")
        
        # Save config
        with open(os.path.join(output_dir, 'feature_config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"\nArtifacts saved to {output_dir}")