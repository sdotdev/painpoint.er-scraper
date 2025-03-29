#!/usr/bin/env python3
"""
Output Manager module for PainPoint.er Search
"""

import json
import os
from typing import Dict, List

import pandas as pd


class OutputManager:
    """Manage output of analysis results."""
    
    @staticmethod
    def save_to_csv(data: Dict, output_path: str):
        """Save analysis results to CSV."""
        # Create DataFrames for each component
        
        # Software mentions
        software_df = pd.DataFrame(data['software_mentions'])
        
        # Pain points - Enhanced with severity and sentiment
        pain_points_rows = []
        for pp_group in data['pain_points']:
            software = pp_group['software']
            for pp in pp_group['pain_points']:
                row = {
                    'software': software,
                    'context': pp['context'],
                    'severity': pp.get('severity', 'unknown')
                }
                
                # Add keywords if available
                if 'keywords' in pp:
                    row['keywords'] = ', '.join(pp['keywords'])
                elif 'keyword' in pp:  # Backward compatibility
                    row['keywords'] = pp['keyword']
                    
                # Add sentiment score if available
                if 'sentiment_score' in pp:
                    row['sentiment_score'] = pp['sentiment_score']
                    
                pain_points_rows.append(row)
        pain_points_df = pd.DataFrame(pain_points_rows)
        
        # Demand metrics - Enhanced with platform percentages and engagement
        demand_rows = []
        for software, metrics in data['demand_metrics'].items():
            row = {
                'software': software,
                'total_mentions': metrics['total_mentions'],
                'demand_score': metrics['demand_score']
            }
            
            # Add platform distribution
            for platform, count in metrics.get('platform_distribution', {}).items():
                row[f'mentions_{platform}'] = count
                
            # Add platform percentages if available
            for platform, percentage in metrics.get('platform_percentages', {}).items():
                row[f'percentage_{platform}'] = percentage
                
            # Add recency metrics if available
            if 'recency_metrics' in metrics:
                for key, value in metrics['recency_metrics'].items():
                    row[f'recency_{key}'] = value
                    
            demand_rows.append(row)
        demand_df = pd.DataFrame(demand_rows)
        
        # Sentiment analysis
        sentiment_rows = []
        for software, sentiment_data in data.get('sentiment_analysis', {}).items():
            row = {
                'software': software,
                'average_sentiment': sentiment_data.get('average_sentiment', 0),
                'sentiment_category': sentiment_data.get('sentiment_category', 'unknown'),
                'sample_size': sentiment_data.get('sample_size', 0)
            }
            
            # Add sentiment distribution if available
            for category, count in sentiment_data.get('sentiment_distribution', {}).items():
                row[f'sentiment_{category}_count'] = count
                
            sentiment_rows.append(row)
        sentiment_df = pd.DataFrame(sentiment_rows) if sentiment_rows else None
        
        # Statistical analysis summary
        stats_rows = []
        if 'statistical_analysis' in data and 'mention_statistics' in data['statistical_analysis']:
            stats = data['statistical_analysis']['mention_statistics']
            stats_row = {
                'mean_mentions': stats.get('mean', 0),
                'median_mentions': stats.get('median', 0),
                'std_dev_mentions': stats.get('standard_deviation', 0),
                'min_mentions': stats.get('min', 0),
                'max_mentions': stats.get('max', 0)
            }
            stats_rows.append(stats_row)
        stats_df = pd.DataFrame(stats_rows) if stats_rows else None
        
        # Trend analysis
        trend_rows = []
        for software, trend_data in data.get('trend_analysis', {}).items():
            row = {
                'software': software,
                'trend_direction': trend_data.get('trend_direction', 'unknown'),
                'trend_slope': trend_data.get('slope', 0),
                'data_points': trend_data.get('data_points', 0)
            }
            trend_rows.append(row)
        trend_df = pd.DataFrame(trend_rows) if trend_rows else None
        
        # Product ideas - Enhanced with more details
        ideas_df = pd.DataFrame(data['product_ideas'])
        
        # Save to separate CSV files
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base_path = os.path.splitext(output_path)[0]
        
        # Save all dataframes to CSV
        csv_files = []
        
        software_df.to_csv(f"{base_path}_software.csv", index=False)
        csv_files.append(f"{base_path}_software.csv")
        
        pain_points_df.to_csv(f"{base_path}_pain_points.csv", index=False)
        csv_files.append(f"{base_path}_pain_points.csv")
        
        demand_df.to_csv(f"{base_path}_demand.csv", index=False)
        csv_files.append(f"{base_path}_demand.csv")
        
        ideas_df.to_csv(f"{base_path}_ideas.csv", index=False)
        csv_files.append(f"{base_path}_ideas.csv")
        
        # Save new analysis dataframes if they exist
        if sentiment_df is not None:
            sentiment_df.to_csv(f"{base_path}_sentiment.csv", index=False)
            csv_files.append(f"{base_path}_sentiment.csv")
            
        if stats_df is not None:
            stats_df.to_csv(f"{base_path}_statistics.csv", index=False)
            csv_files.append(f"{base_path}_statistics.csv")
            
        if trend_df is not None:
            trend_df.to_csv(f"{base_path}_trends.csv", index=False)
            csv_files.append(f"{base_path}_trends.csv")
        
        # Save combined results to main CSV with enhanced information
        combined_rows = []
        for idea in data['product_ideas']:
            row = {
                'software': idea['target_software'],
                'demand_score': idea['demand_score'],
                'idea_description': idea['idea_description']
            }
            
            # Add sentiment if available
            if 'sentiment_analysis' in data and idea['target_software'] in data['sentiment_analysis']:
                sentiment = data['sentiment_analysis'][idea['target_software']]
                row['sentiment'] = sentiment.get('sentiment_category', 'unknown')
                row['sentiment_score'] = sentiment.get('average_sentiment', 0)
            
            # Add trend if available
            if 'trend_analysis' in data and idea['target_software'] in data['trend_analysis']:
                trend = data['trend_analysis'][idea['target_software']]
                row['trend'] = trend.get('trend_direction', 'unknown')
            
            combined_rows.append(row)
            
        combined_df = pd.DataFrame(combined_rows)
        combined_df.to_csv(output_path, index=False)
        csv_files.append(output_path)
        
        return csv_files
    
    @staticmethod
    def save_to_json(data: Dict, output_path: str):
        """Save analysis results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return output_path