#!/usr/bin/env python3
"""
URL Processor module for PainPoint.er Search
"""

import csv
import os
from typing import Dict, List
from urllib.parse import urlparse


class URLProcessor:
    """Process and normalize URLs from CSV file."""
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Ensure URL has proper scheme."""
        if not url.startswith(('http://', 'https://')):
            return f"https://{url}"
        return url
    
    @staticmethod
    def identify_platform(url: str) -> str:
        """Identify the platform from the URL."""
        domain = urlparse(url).netloc.lower()
        
        platform_mapping = {
            'reddit.com': 'reddit',
            'twitter.com': 'twitter',
            'x.com': 'twitter',
            'youtube.com': 'youtube',
            'producthunt.com': 'producthunt',
            'facebook.com': 'facebook',
            'linkedin.com': 'linkedin',
            'github.com': 'github',
            'medium.com': 'medium',
            'quora.com': 'quora',
            'stackoverflow.com': 'stackoverflow',
            'discord.com': 'discord',
            'slack.com': 'slack'
        }
        
        for key, value in platform_mapping.items():
            if key in domain:
                return value
        
        return 'generic'
    
    @classmethod
    def process_csv(cls, csv_path: str) -> List[Dict]:
        """Process CSV file containing URLs."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        urls_data = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            if 'url' not in reader.fieldnames:
                raise ValueError("CSV must contain a 'url' column")
            
            for row in reader:
                url = row['url']
                normalized_url = cls.normalize_url(url)
                platform = cls.identify_platform(normalized_url)
                
                urls_data.append({
                    'original_url': url,
                    'normalized_url': normalized_url,
                    'platform': platform,
                    **{k: v for k, v in row.items() if k != 'url'}
                })
        
        return urls_data