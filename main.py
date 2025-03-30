#!/usr/bin/env python3
"""
PainPoint.er Search - A tool to identify software pain points and product opportunities
by analyzing user feedback from various online communities.
"""

import argparse
import sys
import os
from pathlib import Path

# Import dotenv for environment variable support
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if it exists
    load_dotenv()
    env_loaded = True
except ImportError:
    env_loaded = False
    print("Note: python-dotenv not installed. Environment variables from .env file will not be loaded.")
    print("To enable .env support, install python-dotenv: pip install python-dotenv")

from src.painpointer import PainPointerSearch
from src.console.utils import clear_console, display_logo


def get_input_with_default(prompt, default=None):
    """
    Prompt user for input with a default value shown in greyed-out brackets.
    If user enters nothing, return the default value.
    """
    if default is not None:
        # ANSI escape code for grey text
        grey = '\033[90m'
        reset = '\033[0m'
        prompt = f"{prompt} {grey}[{default}]{reset}: "
    else:
        prompt = f"{prompt}: "
    
    user_input = input(prompt)
    return default if user_input.strip() == '' else user_input


def main():
    """Main entry point for PainPoint.er Search."""
    # Clear the console and display the logo
    clear_console()
    display_logo()
    
    parser = argparse.ArgumentParser(description="PainPoint.er Search - Identify software pain points and product opportunities")
    # Remove the default value so we can check if it was provided or not.
    parser.add_argument('csv_file', nargs='?', default=None, help='CSV file containing URLs to analyze (default: urls.csv)')
    parser.add_argument('-d', '--days', type=int, default=7, help='Number of days to look back (default: 7)')
    parser.add_argument('-o', '--output', default='./output', help='Output directory for results (default: ./output)')
    parser.add_argument('-k', '--api-key', help='API key for AI analysis (optional)')
    parser.add_argument('-a', '--api-type', default='gemini', choices=['gemini', 'openai', 'anthropic', 'deepseek', 'none'], 
                        help='AI API to use for analysis (default: gemini)')
    
    # Parse known args to handle missing arguments interactively
    args, unknown = parser.parse_known_args()
    
    # Prompt for missing CSV file argument
    if args.csv_file is None:
        args.csv_file = get_input_with_default("Enter CSV file containing URLs to analyze", 'urls.csv')
        if not args.csv_file:
            print("Error: CSV file is required")
            sys.exit(1)
    
    # Prompt for days with default value
    if '--days' not in sys.argv and '-d' not in sys.argv:
        days_input = get_input_with_default("Enter number of days to look back", args.days)
        try:
            args.days = int(days_input)
        except ValueError:
            print(f"Error: '{days_input}' is not a valid number of days. Using default: {args.days}")
    
    # Prompt for output directory with default value
    if '--output' not in sys.argv and '-o' not in sys.argv:
        args.output = get_input_with_default("Enter output directory for results", args.output)
    
    # Check for API keys in environment variables
    gemini_key = os.environ.get('GEMINI_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    github_token = os.environ.get('GITHUB_TOKEN')  # For DeepSeek-V3
    
    # Determine default API type based on available keys
    default_api_type = args.api_type  # Start with command-line default
    if not args.api_type or args.api_type == 'gemini':
        if gemini_key:
            default_api_type = 'gemini'
        elif openai_key:
            default_api_type = 'openai'
        elif anthropic_key:
            default_api_type = 'anthropic'
        elif github_token:
            default_api_type = 'deepseek'
    
    # Get default API key based on selected API type
    default_api_key = None
    if default_api_type == 'gemini':
        default_api_key = gemini_key
    elif default_api_type == 'openai':
        default_api_key = openai_key
    elif default_api_type == 'anthropic':
        default_api_key = anthropic_key
    elif default_api_type == 'deepseek':
        default_api_key = github_token
    
    # Prompt for API key if not provided via command line
    if '--api-key' not in sys.argv and '-k' not in sys.argv:
        api_key_prompt = "Enter API key for AI analysis"
        if default_api_key:
            api_key_prompt += f" (found {default_api_type} key in .env)"
        args.api_key = get_input_with_default(api_key_prompt, default_api_key)
    
    # Prompt for API type with default value
    if '--api-type' not in sys.argv and '-a' not in sys.argv:
        api_type_prompt = "Enter AI API to use for analysis"
        if default_api_key:
            api_type_prompt += f" (using {default_api_type} from .env)"
        args.api_type = get_input_with_default(api_type_prompt, default_api_type)
        if args.api_type not in ['gemini', 'openai', 'anthropic', 'deepseek', 'none']:
            print(f"Warning: '{args.api_type}' is not a valid API type. Using default: {default_api_type}")
            args.api_type = default_api_type
    
    try:
        # Create and run PainPointer search
        pain_pointer = PainPointerSearch(
            csv_path=args.csv_file,
            days_back=args.days,
            output_dir=args.output,
            api_key=args.api_key,
            api_type=args.api_type
        )
        
        results = pain_pointer.run()
        
        print("\nPainPoint.er Search completed successfully!")
        print(f"Found {len(results['analysis_results']['software_mentions'])} software mentions")
        print(f"Identified {len(results['analysis_results']['pain_points'])} pain points")
        print(f"Generated {len(results['analysis_results']['product_ideas'])} product ideas")
        
        # Print statistical analysis summary if available
        if 'statistical_analysis' in results['analysis_results'] and 'mention_statistics' in results['analysis_results']['statistical_analysis']:
            stats = results['analysis_results']['statistical_analysis']['mention_statistics']
            print("\nStatistical Analysis:")
            print(f"Mean mentions: {stats.get('mean', 0):.2f}")
            print(f"Median mentions: {stats.get('median', 0)}")
            print(f"Standard deviation: {stats.get('standard_deviation', 0):.2f}")
            
            # Print outliers if available
            if 'outliers' in stats and stats['outliers']:
                print(f"Software with unusually high mentions: {', '.join(stats['outliers'][:3])}")
        
        # Print sentiment analysis summary if available
        if 'sentiment_analysis' in results['analysis_results']:
            sentiment_data = results['analysis_results']['sentiment_analysis']
            if sentiment_data:
                print("\nSentiment Analysis:")
                # Get top 3 most negative and most positive software
                sorted_sentiment = sorted(sentiment_data.items(), 
                                          key=lambda x: x[1].get('average_sentiment', 0))
                
                if len(sorted_sentiment) >= 2:
                    # Most negative
                    software, data = sorted_sentiment[0]
                    print(f"Most negative sentiment: {software} ({data.get('average_sentiment', 0):.2f})")
                    
                    # Most positive
                    software, data = sorted_sentiment[-1]
                    print(f"Most positive sentiment: {software} ({data.get('average_sentiment', 0):.2f})")
        
        # Print trend analysis if available
        if 'trend_analysis' in results['analysis_results'] and results['analysis_results']['trend_analysis']:
            print("\nTrend Analysis:")
            increasing = []
            decreasing = []
            
            for software, trend in results['analysis_results']['trend_analysis'].items():
                if trend.get('trend_direction') == 'increasing':
                    increasing.append(software)
                elif trend.get('trend_direction') == 'decreasing':
                    decreasing.append(software)
            
            if increasing:
                print(f"Increasing trends: {', '.join(increasing[:3])}")
            if decreasing:
                print(f"Decreasing trends: {', '.join(decreasing[:3])}")
        
        # Print top 3 product ideas with enhanced information
        if results['analysis_results']['product_ideas']:
            print("\nTop 3 Product Ideas:")
            for i, idea in enumerate(results['analysis_results']['product_ideas'][:3], 1):
                description = idea['idea_description']
                demand_score = idea['demand_score']
                
                # Add sentiment if available
                sentiment_info = ""
                if 'sentiment_analysis' in results['analysis_results'] and \
                   idea['target_software'] in results['analysis_results']['sentiment_analysis']:
                    sentiment = results['analysis_results']['sentiment_analysis'][idea['target_software']]
                    sentiment_info = f", Sentiment: {sentiment.get('sentiment_category', 'unknown')}"
                
                # Add trend if available
                trend_info = ""
                if 'trend_analysis' in results['analysis_results'] and \
                   idea['target_software'] in results['analysis_results']['trend_analysis']:
                    trend = results['analysis_results']['trend_analysis'][idea['target_software']]
                    trend_info = f", Trend: {trend.get('trend_direction', 'unknown')}"
                
                print(f"{i}. {description} (Demand Score: {demand_score:.2f}{sentiment_info}{trend_info})")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
