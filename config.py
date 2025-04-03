# config.py
import os
from dotenv import load_dotenv

def load_config():
    """Loads configuration from .env file."""
    load_dotenv()
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "azure_github_api_key": os.getenv("AZURE_GITHUB_API_KEY"),
        "reddit_client_id": os.getenv("REDDIT_CLIENT_ID"),
        "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
        "reddit_user_agent": os.getenv("REDDIT_USER_AGENT"),
        "urls_file": "urls.csv",
        "output_dir": "output",
        # Add default values or raise errors if keys are missing
    }
    # Basic validation
    if not all([config["reddit_client_id"], config["reddit_client_secret"], config["reddit_user_agent"]]):
        raise ValueError("Missing Reddit API credentials in .env file.")
    if not config["openai_api_key"] and not config["azure_github_api_key"]:
         print("Warning: No AI API keys found in .env. AI functionality will require a key.")
    # You might add more specific validation depending on the chosen AI

    return config

# Load configuration globally for easy access if needed, or pass it around.
# For simplicity here, we might load it in main.py and pass it.
# CONFIG = load_config()