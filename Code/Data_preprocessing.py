'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-12-01 13:52:46
LastEditors: RemoteScy 98257102+Schuyn@users.noreply.github.com
LastEditTime: 2025-12-02 17:56:59
FilePath: /Unsupervised-Machine-Learning-Final-Project/Code/Data_preprocessing.py
Description: 
    This module handles:
    - Loading raw data
    - Train/validation split by year
    - Missing value imputation
    - Feature categorization
    - Data export for downstream tasks
'''
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List

class NBADataPreprocessor:
    def __init__(self, data_path: str, validation_year: int = 2021):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the CSV file
            validation_year: Year to split for validation set
        """
        self.data_path = data_path
        self.validation_year = validation_year
        self.df_raw = None
        self.df_train = None
        self.df_validation = None
        
        # Feature definitions
        self.meta_cols = ['id']
        self.identifier_cols = ['player', 'year', 'rank', 'overall_pick', 'team', 'college']
        
        self.total_stats = [
            'years_active', 'games', 'minutes_played', 
            'points', 'total_rebounds', 'assists'
        ]
        
        self.percentage_stats = [
            'field_goal_percentage', '3_point_percentage', 'free_throw_percentage'
        ]
        
        self.per_game_stats = [
            'average_minutes_played', 'points_per_game',
            'average_total_rebounds', 'average_assists'
        ]
        
        self.advanced_metrics = [
            'win_shares', 'win_shares_per_48_minutes',
            'box_plus_minus', 'value_over_replacement'
        ]
        
        self.categorical_features = ['year', 'team', 'college']
        
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV
        
        Returns:
            Raw dataframe
        """
        self.df_raw = pd.read_csv(self.data_path)
        
        print(f"Loaded {len(self.df_raw)} players from {self.df_raw['year'].min()} to {self.df_raw['year'].max()}")
        print(f"Total columns: {len(self.df_raw.columns)}")
        
        return self.df_raw
    
    def split_train_validation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets by year
        
        Returns:
            Tuple of (train_df, validation_df)
        """
        self.df_train = self.df_raw[self.df_raw['year'] < self.validation_year].copy()
        self.df_validation = self.df_raw[self.df_raw['year'] == self.validation_year].copy()
        
        print(f"Training set: {len(self.df_train)} players ({self.df_train['year'].min()}-{self.df_train['year'].max()})")
        print(f"Validation set: {len(self.df_validation)} players ({self.validation_year})")
        
        return self.df_train, self.df_validation
    
    def analyze_missing_values(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
        """
        Analyze missing value patterns
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name for display
            
        Returns:
            Missing value statistics
        """
        missing_stats = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2),
            'Data_Type': df.dtypes.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        missing_stats = missing_stats[missing_stats['Missing_Count'] > 0]
        
        if len(missing_stats) > 0:
            print(missing_stats.to_string(index=False))
        else:
            print("No missing values found!")
        
        # Analyze players with no career data
        career_cols = ['games', 'minutes_played', 'points']
        no_career = df[career_cols].isnull().all(axis=1).sum()
        print(f"\nPlayers with NO career data: {no_career} ({no_career/len(df)*100:.2f}%)")
        
        return missing_stats
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'preserve_busts') -> pd.DataFrame:
        """
        Handle missing values based on specified strategy
        
        Args:
            df: DataFrame to process
            strategy: 'preserve_busts' (keep players with no games) or 'drop_busts' (remove them)
            
        Returns:
            Processed dataframe
        """
        df_processed = df.copy()
        
        # Step 1: Identify bust players (no games played)
        bust_mask = df_processed['games'].isnull()
        n_busts = bust_mask.sum()
        
        # Step 2: Apply bust handling strategy
        if strategy == 'drop_busts':
            df_processed = df_processed[~bust_mask].copy()
            print(f"Dropped {n_busts} players with no games played")
        elif strategy == 'preserve_busts':
            print(f"Preserved {n_busts} players with no games played")
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'preserve_busts' or 'drop_busts'")
        
        # Step 3: Fill missing values uniformly for all remaining players
        # (applies to both strategies - either bust players or players with partial data)
        
        # Fill total stats with 0
        for col in self.total_stats:
            df_processed[col] = df_processed[col].fillna(0)
        
        # Fill percentage stats with 0 (no attempts = 0%)
        for col in self.percentage_stats:
            df_processed[col] = df_processed[col].fillna(0)
        
        # Fill per-game stats with 0
        for col in self.per_game_stats:
            df_processed[col] = df_processed[col].fillna(0)
        
        # Fill advanced metrics with 0 (no contribution)
        for col in self.advanced_metrics:
            df_processed[col] = df_processed[col].fillna(0)
        
        # Fill missing college with 'International'
        df_processed['college'] = df_processed['college'].fillna('International')
        
        print(f"Final dataset size: {len(df_processed)} players")
        
        return df_processed
    
    def separate_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Separate features into different categories for different purposes
        
        Args:
            df: DataFrame to separate
            
        Returns:
            Dictionary with separated feature groups
        """
        # Drop meta columns
        df_working = df.drop(columns=self.meta_cols, errors='ignore')
        
        # Create feature dictionaries
        feature_groups = {
            'identifiers': df_working[['player', 'year', 'overall_pick', 'team', 'college']],
            'modeling_numeric': df_working[self.total_stats + self.percentage_stats],
            'display_per_game': df_working[self.per_game_stats] if all(col in df_working.columns for col in self.per_game_stats) else None,
            'display_advanced': df_working[self.advanced_metrics] if all(col in df_working.columns for col in self.advanced_metrics) else None,
            'categorical': df_working[self.categorical_features]
        }
        
        print("Feature groups created:")
        for group_name, group_df in feature_groups.items():
            if group_df is not None:
                print(f"  - {group_name}: {len(group_df.columns)} features")
        
        return feature_groups
    
    def save_processed_data(self, output_dir: str = 'Data/processed'):
        """
        Save processed data to files
        
        Args:
            output_dir: Directory to save processed data
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train/validation splits
        train_path = os.path.join(output_dir, 'train_data.csv')
        val_path = os.path.join(output_dir, 'validation_data.csv')
        
        self.df_train.to_csv(train_path, index=False)
        self.df_validation.to_csv(val_path, index=False)
        
        print(f"Saved training data: {train_path}")
        print(f"Saved validation data: {val_path}")
        
        print("\nData preprocessing complete!")
    
    def run_pipeline(self, missing_value_strategy: str = 'preserve_busts') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline
        
        Args:
            missing_value_strategy: Strategy for handling missing values
            
        Returns:
            Tuple of (processed_train, processed_validation)
        """
        # Load data
        self.load_data()
        
        # Split train/validation
        self.split_train_validation()
        
        # Analyze missing values
        self.analyze_missing_values(self.df_train, "Training Set")
        self.analyze_missing_values(self.df_validation, "Validation Set")
        
        # Handle missing values
        self.df_train = self.handle_missing_values(self.df_train, strategy=missing_value_strategy)
        self.df_validation = self.handle_missing_values(self.df_validation, strategy=missing_value_strategy)
        
        # Save processed data
        self.save_processed_data()
        
        return self.df_train, self.df_validation


def main():
    """
    Main execution function for testing
    """
    # Initialize preprocessor
    preprocessor = NBADataPreprocessor(
        data_path='Data/nbaplayersdraft.csv',
        validation_year=2021
    )
    
    # Run pipeline
    df_train, df_validation = preprocessor.run_pipeline(missing_value_strategy='preserve_busts')
    
    # Display summary
    print("PREPROCESSING SUMMARY")
    print(f"Training set shape: {df_train.shape}")
    print(f"Validation set shape: {df_validation.shape}")
    print("\nTraining set columns:")
    print(df_train.columns.tolist())


if __name__ == "__main__":
    main()