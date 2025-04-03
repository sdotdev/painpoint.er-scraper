# main.py
import sys
from config import load_config
from utils import read_urls, is_reddit_url, save_results, ensure_dir
from scraper import RedditScraper # Import specific scrapers needed
from ai_clients import get_ai_client # Import the factory function
from analysis import perform_analysis

def main():
    """Main function to run the scraping and analysis process."""
    try:
        cfg = load_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)

    # --- User Choice for AI Provider ---
    print("Choose your AI provider:")
    print("1: OpenAI")
    print("2: Azure (using GitHub Models API setup)")
    # Add more options here if you add more clients

    ai_choice = input("Enter the number of your choice: ")

    selected_provider_name = None
    if ai_choice == '1':
        selected_provider_name = "openai"
        if not cfg.get("openai_api_key"):
            print("Error: OpenAI API key not found in .env file.")
            sys.exit(1)
    elif ai_choice == '2':
        selected_provider_name = "azure"
        if not cfg.get("azure_github_api_key"):
             print("Error: Azure/GitHub API key (AZURE_GITHUB_API_KEY) not found in .env file.")
             sys.exit(1)
    else:
        print("Invalid choice.")
        sys.exit(1)

    # --- Initialize AI Client ---
    try:
        ai_client = get_ai_client(selected_provider_name, cfg)
    except (ValueError, ImportError, Exception) as e:
        print(f"Error initializing AI client '{selected_provider_name}': {e}")
        sys.exit(1)

    # --- Read URLs ---
    urls_to_scrape = read_urls(cfg['urls_file'])
    if not urls_to_scrape:
        print("No URLs found in the CSV file or file could not be read. Exiting.")
        sys.exit(1)

    print(f"Found {len(urls_to_scrape)} URLs to process from {cfg['urls_file']}")

    # --- Scraping ---
    all_scraped_comments = []
    reddit_scraper = None # Initialize scraper lazily

    for item in urls_to_scrape:
        url = item.get('url', '').strip()
        if not url:
            continue

        print(f"\nProcessing URL: {url}")

        # --- URL Type Handling (Modular Part) ---
        if is_reddit_url(url):
            # Initialize Reddit scraper only if needed and not already done
            if reddit_scraper is None:
                try:
                    reddit_scraper = RedditScraper(
                        client_id=cfg['reddit_client_id'],
                        client_secret=cfg['reddit_client_secret'],
                        user_agent=cfg['reddit_user_agent']
                    )
                except (ValueError, Exception) as e:
                     print(f"Fatal Error: Could not initialize Reddit scraper: {e}. Skipping all Reddit URLs.")
                     # Optionally break or continue depending on desired behavior
                     break # Stop processing if Reddit scraper fails fundamentally

            # Scrape if scraper is available
            if reddit_scraper:
                 try:
                     comments = reddit_scraper.scrape_comments(url, limit=200) # Adjust limit as needed
                     all_scraped_comments.extend(comments)
                 except Exception as e: # Catch errors during scraping for a specific URL
                      print(f"An error occurred while scraping {url}: {e}")
            else:
                 print("Skipping Reddit URL as scraper initialization failed earlier.")

        # --- Add other URL handlers here ---
        # elif is_twitter_url(url):
        #    # twitter_scraper = get_twitter_scraper(...)
        #    # content = twitter_scraper.scrape(...)
        #    # all_scraped_content.extend(content)
        #    pass
        else:
            print(f"Skipping unsupported URL type: {url}")
            # For now, just skip non-Reddit URLs as requested

    # --- Analysis ---
    if not all_scraped_comments:
        print("\nNo comments were successfully scraped. Cannot perform analysis.")
        sys.exit(0) # Exit gracefully

    print(f"\nTotal comments scraped: {len(all_scraped_comments)}")
    print("Starting analysis...")

    analysis_result = perform_analysis(ai_client, all_scraped_comments)

    # --- Save Results ---
    print("\nAnalysis complete.")
    output_filename_prefix = f"{selected_provider_name}_analysis"
    save_results(cfg['output_dir'], output_filename_prefix, analysis_result)

    print("\nProcess finished.")

if __name__ == "__main__":
    main()