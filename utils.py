# utils.py
import os
import csv
from urllib.parse import urlparse
from datetime import datetime

def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)

def save_results(output_dir, filename_prefix, data):
    """Saves the analysis results to a timestamped file."""
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(data))
        print(f"Results saved successfully to: {filepath}")
    except IOError as e:
        print(f"Error saving results to {filepath}: {e}")

def read_urls(filepath):
    """Reads URLs from a CSV file."""
    urls_data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                urls_data.append(row) # row is {'url': '...', 'category': '...', 'notes': '...'}
    except FileNotFoundError:
        print(f"Error: URLs file not found at {filepath}")
    except Exception as e:
        print(f"Error reading CSV file {filepath}: {e}")
    return urls_data

def is_reddit_url(url_string):
    """Checks if the URL is a Reddit URL."""
    try:
        parsed_url = urlparse(url_string)
        return parsed_url.netloc.lower().endswith('reddit.com')
    except Exception:
        return False # Handle potential parsing errors

def extract_subreddit_name(reddit_url):
    """Extracts the subreddit name from a Reddit URL."""
    try:
        path_parts = urlparse(reddit_url).path.strip('/').split('/')
        if len(path_parts) >= 2 and path_parts[0] == 'r':
            return path_parts[1]
    except Exception:
        pass # Handle parsing errors
    return None