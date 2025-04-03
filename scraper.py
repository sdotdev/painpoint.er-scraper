# scraper.py
import praw
from utils import extract_subreddit_name

class RedditScraper:
    """Scrapes comments from Reddit subreddits."""
    def __init__(self, client_id, client_secret, user_agent):
        if not all([client_id, client_secret, user_agent]):
            raise ValueError("Missing Reddit API credentials for PRAW.")
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                read_only=True # Read-only mode is sufficient for scraping comments
            )
            print("PRAW Reddit client initialized successfully.")
        except Exception as e:
            print(f"Error initializing PRAW Reddit client: {e}")
            raise # Re-raise the exception to stop execution if PRAW fails

    def scrape_comments(self, subreddit_url, limit=100):
        """
        Scrapes comments from a given subreddit URL.

        Args:
            subreddit_url (str): The URL of the subreddit (e.g., https://reddit.com/r/productivity).
            limit (int): The maximum number of comments to fetch.

        Returns:
            list[str]: A list of comment bodies (text).
        """
        subreddit_name = extract_subreddit_name(subreddit_url)
        if not subreddit_name:
            print(f"Could not extract subreddit name from URL: {subreddit_url}")
            return []

        comments_text = []
        try:
            print(f"Scraping comments from r/{subreddit_name} (limit: {limit})...")
            subreddit = self.reddit.subreddit(subreddit_name)
            # Fetching comments (can be slow)
            # .hot(), .new(), .top(), .controversial() are also options instead of .comments
            for comment in subreddit.comments(limit=limit):
                 # Skip potentially deleted or empty comments
                if hasattr(comment, 'body') and comment.body and comment.body != '[deleted]' and comment.body != '[removed]':
                    comments_text.append(comment.body)
            print(f"Scraped {len(comments_text)} comments from r/{subreddit_name}.")
        except praw.exceptions.PRAWException as e:
            print(f"PRAW error scraping r/{subreddit_name}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred scraping r/{subreddit_name}: {e}")

        return comments_text

# --- Placeholder for future scraper types ---
# class OtherWebsiteScraper:
#     def __init__(self, config):
#         # Initialization for another type of scraper (e.g., using requests/BeautifulSoup)
#         pass
#
#     def scrape_content(self, url):
#         # Logic to scrape content from a different website structure
#         print(f"Placeholder: Would scrape {url} using a different method.")
#         return ["Example content from other site."]