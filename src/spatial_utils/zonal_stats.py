#!/usr/bin/env python3
"""
Zonal Statistics Script for Raster Data Analysis

This script calculates zonal statistics for raster data using polygon boundaries.
It supports multiple administrative levels and various statistical measures.

Usage:
    python zonal_statistics.py --raster path/to/raster.tif --shapefile path/to/shapefile.shp --admin_level NAME_1 --stats mean median std
"""

import argparse
import sys
import os
import warnings
import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class ZonalStatistics:
    """Class to handle zonal statistics calculations"""
    
    def __init__(self, raster_path, shapefile_path, admin_level='NAME_1'):
        """
        Initialize ZonalStatistics object
        
        Parameters:
        -----------
        raster_path : str
            Path to the raster file
        shapefile_path : str
            Path to the shapefile
        admin_level : str
            Column name for administrative level (default: 'NAME_1')
        """
        self.raster_path = raster_path
        self.shapefile_path = shapefile_path
        self.admin_level = admin_level
        self.raster = None
        self.shapefile = None
        
    def load_data(self):
        """Load raster and shapefile data"""
        try:
            # Load raster
            self.raster = rasterio.open(self.raster_path)
            print(f"✓ Loaded raster: {self.raster_path}")
            print(f"  - Shape: {self.raster.width} x {self.raster.height}")
            print(f"  - CRS: {self.raster.crs}")
            print(f"  - Bands: {self.raster.count}")
            
            # Load shapefile
            self.shapefile = gpd.read_file(self.shapefile_path)
            print(f"✓ Loaded shapefile: {self.shapefile_path}")
            print(f"  - Features: {len(self.shapefile)}")
            print(f"  - CRS: {self.shapefile.crs}")
            print(f"  - Columns: {list(self.shapefile.columns)}")
            
            # Check if admin_level exists
            if self.admin_level not in self.shapefile.columns:
                available_cols = [col for col in self.shapefile.columns if col != 'geometry']
                raise ValueError(f"Admin level '{self.admin_level}' not found. Available columns: {available_cols}")
                
            # Reproject shapefile to match raster CRS if needed
            if self.shapefile.crs != self.raster.crs:
                print(f"! Reprojecting shapefile from {self.shapefile.crs} to {self.raster.crs}")
                self.shapefile = self.shapefile.to_crs(self.raster.crs)
                
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            sys.exit(1)
    
    def calculate_statistics(self, stats_list=['mean', 'median', 'std', 'min', 'max', 'count']):
        """
        Calculate zonal statistics for each polygon
        
        Parameters:
        -----------
        stats_list : list
            List of statistics to calculate. Options: 'mean', 'median', 'std', 'min', 'max', 'count', 'sum'
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with calculated statistics
        """
        results = []
        total_features = len(self.shapefile)
        
        print(f"\nCalculating statistics for {total_features} features...")
        print(f"Statistics to calculate: {', '.join(stats_list)}")
        
        for idx, feature in self.shapefile.iterrows():
            try:
                # Extract raster values for this polygon
                masked_data, masked_transform = mask(
                    self.raster, 
                    [feature.geometry], 
                    crop=True,
                    nodata=self.raster.nodata
                )
                
                # Handle different band counts
                if self.raster.count == 1:
                    values = masked_data[0]  # Single band
                else:
                    values = masked_data.mean(axis=0)  # Multi-band average
                
                # Remove nodata values
                if self.raster.nodata is not None:
                    valid_values = values[values != self.raster.nodata]
                else:
                    valid_values = values[~np.isnan(values)]
                
                # Initialize stats dictionary
                stats = {
                    'admin_name': feature[self.admin_level],
                    'admin_level': self.admin_level,
                    'feature_id': idx
                }
                
                # Calculate requested statistics
                if len(valid_values) > 0:
                    if 'mean' in stats_list:
                        stats['mean'] = np.mean(valid_values)
                    if 'median' in stats_list:
                        stats['median'] = np.median(valid_values)
                    if 'std' in stats_list:
                        stats['std'] = np.std(valid_values)
                    if 'min' in stats_list:
                        stats['min'] = np.min(valid_values)
                    if 'max' in stats_list:
                        stats['max'] = np.max(valid_values)
                    if 'count' in stats_list:
                        stats['count'] = len(valid_values)
                    if 'sum' in stats_list:
                        stats['sum'] = np.sum(valid_values)
                else:
                    # No valid pixels
                    for stat in stats_list:
                        stats[stat] = np.nan
                    if 'count' in stats_list:
                        stats['count'] = 0
                
                results.append(stats)
                
                # Progress indicator
                if (idx + 1) % max(1, total_features // 10) == 0:
                    print(f"  Processed {idx + 1}/{total_features} features ({(idx + 1)/total_features*100:.1f}%)")
                    
            except Exception as e:
                print(f"  Warning: Error processing feature {idx} ({feature[self.admin_level]}): {e}")
                # Add empty record
                stats = {
                    'admin_name': feature[self.admin_level],
                    'admin_level': self.admin_level,
                    'feature_id': idx
                }
                for stat in stats_list:
                    stats[stat] = np.nan
                if 'count' in stats_list:
                    stats['count'] = 0
                results.append(stats)
        
        df = pd.DataFrame(results)
        print(f"✓ Completed zonal statistics calculation")
        return df
    
    def create_summary_plot(self, stats_df, output_path=None):
        """
        Create summary visualizations of the statistics
        
        Parameters:
        -----------
        stats_df : pandas.DataFrame
            DataFrame with calculated statistics
        output_path : str, optional
            Path to save the plot
        """
        # Filter numeric columns
        numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['feature_id', 'count']]
        
        if len(numeric_cols) == 0:
            print("No numeric columns found for plotting")
            return
        
        # Create subplots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                valid_data = stats_df[col].dropna()
                if len(valid_data) > 0:
                    axes[i].hist(valid_data, bins=min(20, len(valid_data)), alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col.title()}')
                    axes[i].set_xlabel(col.title())
                    axes[i].set_ylabel('Frequency')
                else:
                    axes[i].text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Distribution of {col.title()}')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {output_path}")
        
        plt.show()
    
    def save_results(self, stats_df, output_path):
        """
        Save results to CSV file
        
        Parameters:
        -----------
        stats_df : pandas.DataFrame
            DataFrame with calculated statistics
        output_path : str
            Path to save the CSV file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to CSV
            stats_df.to_csv(output_path, index=False)
            print(f"✓ Results saved to: {output_path}")
            
            # Print summary
            print(f"\nSummary:")
            print(f"  - Total features processed: {len(stats_df)}")
            print(f"  - Features with valid data: {stats_df['count'].gt(0).sum() if 'count' in stats_df.columns else 'N/A'}")
            
        except Exception as e:
            print(f"✗ Error saving results: {e}")

