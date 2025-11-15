import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

class EDA:
    def __init__(self, df: pd.DataFrame):
        """Initialize EDA with a DataFrame."""
        self.df = df
        self.report = {}
    
    def generate_summary(self) -> Dict:
        """Generate a summary of the dataset."""
        summary = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'descriptive_stats': self.df.describe(include='all').to_dict()
        }
        self.report['summary'] = summary
        return summary
    
    def plot_missing_values(self, save_path: Optional[str] = None) -> None:
        """Plot missing values heatmap."""
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_numeric_distributions(self, columns: List[str] = None, save_dir: str = None) -> None:
        """Plot distributions of numeric columns."""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in self.df.columns]
        
        for col in numeric_cols:
            plt.figure(figsize=(10, 4))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
            
            if save_dir:
                save_path = Path(save_dir) / f'distribution_{col}.png'
                plt.savefig(save_path, bbox_inches='tight')
            plt.close()
    
    def plot_categorical_counts(self, column: str, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """Plot value counts for a categorical column."""
        if column not in self.df.columns:
            print(f"Column {column} not found in DataFrame.")
            return
        
        value_counts = self.df[column].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=value_counts.values, y=value_counts.index)
        plt.title(f'Top {top_n} {column} by Count')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_geo_distribution(self, location_col: str = 'state', 
                            value_col: str = 'main_workers_total',
                            save_path: Optional[str] = None) -> go.Figure:
        """Create a choropleth map of the data."""
        if location_col not in self.df.columns or value_col not in self.df.columns:
            print(f"Required columns not found in DataFrame.")
            return None
        
        # Group by location and sum the values
        geo_df = self.df.groupby(location_col)[value_col].sum().reset_index()
        
        # Create choropleth
        fig = px.choropleth(
            geo_df,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations=location_col,
            color=value_col,
            color_continuous_scale='Viridis',
            title=f'Distribution of {value_col} by State'
        )
        
        fig.update_geos(
            fitbounds="locations",
            visible=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_correlation_heatmap(self, columns: List[str] = None, 
                                   save_path: Optional[str] = None) -> None:
        """Generate a correlation heatmap for numeric columns."""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in self.df.columns]
        
        if len(numeric_cols) < 2:
            print("Not enough numeric columns for correlation analysis.")
            return
        
        corr = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
        plt.title('Correlation Heatmap')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_dir: str = 'reports') -> None:
        """Generate a comprehensive EDA report."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all plots and save them
        self.plot_missing_values(os.path.join(output_dir, 'missing_values.png'))
        
        # Generate and save numeric distributions
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            self.plot_numeric_distributions([col], output_dir)
        
        # Generate and save categorical plots
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            self.plot_categorical_counts(col, save_path=os.path.join(output_dir, f'categorical_{col}.png'))
        
        # Generate correlation heatmap
        self.generate_correlation_heatmap(save_path=os.path.join(output_dir, 'correlation_heatmap.png'))
        
        # Generate geo distribution
        if 'state' in self.df.columns and 'main_workers_total' in self.df.columns:
            self.plot_geo_distribution(save_path=os.path.join(output_dir, 'geo_distribution.html'))
        
        print(f"EDA report generated in {output_dir}")
