#!/usr/bin/env python3
"""
PainPointer Search - Main module
"""

import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from tqdm import tqdm

from src.url_processor import URLProcessor
from src.scrapers import ScraperFactory
from src.data_analyzer import DataAnalyzer
from src.output_manager import OutputManager


class PainPointerSearch:
    """Main class for PainPoint.er Search tool."""
    
    def __init__(self, csv_path: str, days_back: int, output_dir: str, api_key: str = None, api_type: str = 'gemini'):
        self.csv_path = csv_path
        self.days_back = days_back
        self.output_dir = output_dir
        self.api_key = api_key
        self.api_type = api_type
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self):
        """Run the PainPoint.er Search process."""
        print(f"Starting PainPoint.er Search with {self.days_back} days lookback")
        
        # Process CSV file
        print("Processing URL CSV file...")
        url_processor = URLProcessor()
        urls_data = url_processor.process_csv(self.csv_path)
        print(f"Found {len(urls_data)} URLs to process")
        
        # Scrape data
        print("Scraping data from URLs...")
        scraped_data = self._scrape_urls(urls_data)
        
        # Save raw scraped data
        raw_data_path = os.path.join(self.output_dir, 'raw_data.json')
        with open(raw_data_path, 'w', encoding='utf-8') as f:
            json.dump(scraped_data, f, indent=2)
        print(f"Raw scraped data saved to {raw_data_path}")
        
        # Analyze data
        print("Analyzing scraped data...")
        analyzer = DataAnalyzer(api_key=self.api_key, api_type=self.api_type)
        analysis_results = analyzer.analyze_data(scraped_data)
        
        # Generate AI analysis if API key provided
        if self.api_key:
            print("Generating AI analysis...")
            ai_analysis = analyzer.generate_ai_analysis(analysis_results)
            analysis_results['ai_analysis'] = ai_analysis
            print("Generated AI analysis with qualitative insights")
            
            # Print a summary of the AI insights if available
            if isinstance(ai_analysis, dict) and not ai_analysis.get('error'):
                print("\nAI Analysis Highlights:")
                for section, content in ai_analysis.items():
                    if isinstance(content, str) and section != 'error':
                        # Print first 100 characters of each section
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"- {section.replace('_', ' ').title()}: {preview}")
                print("\nFull AI analysis available in the output JSON file.")
            elif isinstance(ai_analysis, str):
                print(f"\nAI Analysis Preview: {ai_analysis[:100]}...")
                print("Full AI analysis available in the output JSON file.")
        
        # Save results
        print("Saving analysis results...")
        output_manager = OutputManager()
        
        csv_path = os.path.join(self.output_dir, 'painpointer_results.csv')
        csv_files = output_manager.save_to_csv(analysis_results, csv_path)
        print(f"CSV results saved to: {', '.join(csv_files)}")
        
        json_path = os.path.join(self.output_dir, 'painpointer_results.json')
        json_file = output_manager.save_to_json(analysis_results, json_path)
        print(f"JSON results saved to: {json_file}")
        
        return {
            'csv_files': csv_files,
            'json_file': json_file,
            'analysis_results': analysis_results
        }
    
    def _scrape_urls(self, urls_data: List[Dict]) -> List[Dict]:
        """Scrape data from URLs using appropriate scrapers."""
        scraped_data = []
        
        # Use ThreadPoolExecutor for parallel scraping
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for url_data in urls_data:
                url = url_data['normalized_url']
                platform = url_data['platform']
                
                # Get appropriate scraper for platform
                scraper = ScraperFactory.get_scraper(platform, self.days_back)
                
                # Submit scraping task to executor
                future = executor.submit(scraper.scrape, url)
                futures.append((future, url_data))
            
            # Process results as they complete
            for future, url_data in tqdm(futures, desc="Scraping URLs"):
                try:
                    result = future.result()
                    # Add original URL data to result
                    result.update({
                        'original_url_data': url_data
                    })
                    scraped_data.append(result)
                except Exception as e:
                    print(f"Error scraping {url_data['normalized_url']}: {str(e)}")
                    scraped_data.append({
                        'url': url_data['normalized_url'],
                        'platform': url_data['platform'],
                        'error': str(e),
                        'original_url_data': url_data,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
        
        return scraped_data