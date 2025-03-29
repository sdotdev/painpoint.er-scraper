#!/usr/bin/env python3
"""
Scrapers module for PainPoint.er Search
"""

import datetime
from typing import Dict

import requests
from bs4 import BeautifulSoup


class BaseScraper:
    """Base class for all scrapers."""
    
    def __init__(self, days_back: int):
        self.days_back = days_back
        self.cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape(self, url: str) -> Dict:
        """Scrape data from URL."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _is_within_timeframe(self, date_str: str, date_format: str) -> bool:
        """Check if date is within specified timeframe."""
        try:
            date = datetime.datetime.strptime(date_str, date_format)
            return date >= self.cutoff_date
        except ValueError:
            return False
    
    def _extract_text(self, element) -> str:
        """Extract text from BeautifulSoup element."""
        if element:
            return element.get_text(strip=True)
        return ""


class GenericScraper(BaseScraper):
    """Generic scraper for unknown platforms."""
    
    def scrape(self, url: str) -> Dict:
        """Scrape generic website."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract all text content
            all_text = soup.get_text(separator=' ', strip=True)
            
            # Extract all links
            links = [a.get('href') for a in soup.find_all('a', href=True)]
            
            return {
                'url': url,
                'platform': 'generic',
                'title': self._extract_text(soup.title),
                'content': all_text[:5000],  # Limit content size
                'links': links[:100],  # Limit number of links
                'timestamp': datetime.datetime.now().isoformat(),
                'raw_html': response.text[:10000]  # Store partial HTML for further processing
            }
            
        except Exception as e:
            return {
                'url': url,
                'platform': 'generic',
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }


class RedditScraper(BaseScraper):
    """Scraper for Reddit."""
    
    def scrape(self, url: str) -> Dict:
        """Scrape Reddit posts and comments."""
        # Convert to old.reddit.com for easier scraping
        url = url.replace('reddit.com', 'old.reddit.com')
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract post title
            title = self._extract_text(soup.select_one('.title'))
            
            # Extract post content
            post_content = self._extract_text(soup.select_one('.usertext-body'))
            
            # Extract comments
            comments = []
            for comment in soup.select('.comment'):
                comment_text = self._extract_text(comment.select_one('.usertext-body'))
                comment_score = self._extract_text(comment.select_one('.score'))
                comment_time = self._extract_text(comment.select_one('.live-timestamp'))
                
                comments.append({
                    'text': comment_text,
                    'score': comment_score,
                    'time': comment_time
                })
            
            return {
                'url': url,
                'platform': 'reddit',
                'title': title,
                'post_content': post_content,
                'comments': comments,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'url': url,
                'platform': 'reddit',
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }


class TwitterScraper(BaseScraper):
    """Scraper for Twitter/X."""
    
    def scrape(self, url: str) -> Dict:
        """Scrape Twitter posts and replies."""
        try:
            # Note: Twitter requires more sophisticated approaches like using their API
            # or browser automation. This is a simplified example.
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract tweet content (simplified)
            tweet_content = self._extract_text(soup.select_one('[data-testid="tweetText"]'))
            
            # Extract tweet metadata (simplified)
            timestamp = self._extract_text(soup.select_one('[data-testid="tweet"] time'))
            
            # Extract replies (simplified)
            replies = []
            for reply in soup.select('[data-testid="tweet"]')[1:]:  # Skip the first tweet (original)
                reply_text = self._extract_text(reply.select_one('[data-testid="tweetText"]'))
                reply_time = self._extract_text(reply.select_one('time'))
                
                replies.append({
                    'text': reply_text,
                    'time': reply_time
                })
            
            return {
                'url': url,
                'platform': 'twitter',
                'tweet_content': tweet_content,
                'timestamp': timestamp,
                'replies': replies,
                'scrape_time': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'url': url,
                'platform': 'twitter',
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }


class YouTubeScraper(BaseScraper):
    """Scraper for YouTube."""
    
    def scrape(self, url: str) -> Dict:
        """Scrape YouTube video comments."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract video title
            title = self._extract_text(soup.select_one('meta[name="title"]'))
            
            # Extract video description
            description = self._extract_text(soup.select_one('meta[name="description"]'))
            
            # Note: YouTube comments are loaded dynamically with JavaScript
            # A more sophisticated approach using Selenium or YouTube API would be needed
            # This is a simplified placeholder
            
            return {
                'url': url,
                'platform': 'youtube',
                'title': title,
                'description': description,
                'comments': [],  # Would require more sophisticated scraping
                'timestamp': datetime.datetime.now().isoformat(),
                'note': 'YouTube comments require JavaScript rendering or API access'
            }
            
        except Exception as e:
            return {
                'url': url,
                'platform': 'youtube',
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }


class ProductHuntScraper(BaseScraper):
    """Scraper for ProductHunt."""
    
    def scrape(self, url: str) -> Dict:
        """Scrape ProductHunt product and comments."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract product name
            product_name = self._extract_text(soup.select_one('h1'))
            
            # Extract product description
            description = self._extract_text(soup.select_one('h2'))
            
            # Extract comments (simplified)
            comments = []
            for comment in soup.select('.comment'):
                comment_text = self._extract_text(comment)
                comments.append({
                    'text': comment_text
                })
            
            return {
                'url': url,
                'platform': 'producthunt',
                'product_name': product_name,
                'description': description,
                'comments': comments,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'url': url,
                'platform': 'producthunt',
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }


class ScraperFactory:
    """Factory to create appropriate scraper based on platform."""
    
    @staticmethod
    def get_scraper(platform: str, days_back: int):
        """Return appropriate scraper instance based on platform."""
        scrapers = {
            'reddit': RedditScraper(days_back),
            'twitter': TwitterScraper(days_back),
            'youtube': YouTubeScraper(days_back),
            'producthunt': ProductHuntScraper(days_back),
            'generic': GenericScraper(days_back)
            # Add more platform scrapers as needed
        }
        
        return scrapers.get(platform, scrapers['generic'])