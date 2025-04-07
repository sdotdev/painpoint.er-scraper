# PainPoint.er Scraper 🔍

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Discover Software Pain Points & Product Opportunities

PainPoint.er Scraper is an open-source tool that helps entrepreneurs, product managers, and developers identify software pain points and product opportunities by analyzing user feedback from Reddit communities.

### 🚀 Key Features

- **Platform Analysis**: Scrape and analyze user feedback from Reddit communities, and more platforms being added soon
- **Pain Point Identification**: Automatically detect user complaints and frustrations with existing software
- **AI-Powered Insights**: Leverage OpenAI or Azure/GitHub Models API for analysis
- **Modular Design**: Easily extendable architecture for adding more data sources in the future
- **Simple Reporting**: Generate analysis results with actionable insights

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Advanced Configuration](#advanced-configuration)
- [AI Integration](#ai-integration)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/sdotdev/painpointer-search.git
cd painpointer-search
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Copy `.env.example` to `.env` and add your API keys:
   - Reddit API credentials (required)
   - OpenAI API key or Azure/GitHub API key (at least one required)

2. Run the application:

```bash
python main.py
```

This will:
1. Load your configuration from `.env`
2. Prompt you to choose an AI provider (OpenAI or Azure)
3. Process URLs from `urls.csv`
4. Scrape comments from Reddit subreddits
5. Analyze the comments for pain points and opportunities
6. Save results to the `./output` directory

## 💡 Usage Examples

### Basic Usage

```bash
python main.py
```

### Customizing the Configuration

Edit the `.env` file to customize your configuration:

```
# Reddit API Credentials (Required)
REDDIT_CLIENT_ID="your_reddit_client_id"
REDDIT_CLIENT_SECRET="your_reddit_client_secret"
REDDIT_USER_AGENT="your_reddit_user_agent"

# AI API Keys (At least one required)
OPENAI_API_KEY="your_openai_api_key"
AZURE_GITHUB_API_KEY="your_azure_github_api_key"
```

### Customizing URLs to Scrape

Edit the `urls.csv` file to specify which subreddits to analyze:

```
url,category,notes
https://reddit.com/r/productivity,productivity,Popular productivity subreddit
https://reddit.com/r/software,software,Discussions about various software
```

## 📄 Input Format

The input CSV file must contain a column named `url` with the Reddit URLs to analyze. Additional columns are preserved for reference.

Example `urls.csv`:

```
url,category,notes
https://reddit.com/r/productivity,productivity,Popular productivity subreddit
https://reddit.com/r/software,software,Discussions about various software
https://reddit.com/r/SaaS,saas,Software as a Service discussions
```

**Note:** Currently, only Reddit URLs are supported. Non-Reddit URLs in the CSV will be skipped.

## 📊 Output Format

PainPoint.er Scraper generates a single output file in the `output` directory:

```
[ai_provider]_analysis_[timestamp].txt
```

For example: `openai_analysis_20230615_123045.txt`

This file contains the analysis results from the AI provider, including identified:
- Product opportunities
- Feature requests
- Pain points
- New tool suggestions

## ⚙️ Advanced Configuration

### Configuration Options

The application uses a configuration loaded from the `.env` file and the `config.py` module. You can modify these settings:

| Setting | Description | Location |
|---------|-------------|----------|
| `reddit_client_id` | Reddit API client ID | .env file |
| `reddit_client_secret` | Reddit API client secret | .env file |
| `reddit_user_agent` | Reddit API user agent | .env file |
| `openai_api_key` | OpenAI API key | .env file |
| `azure_github_api_key` | Azure/GitHub Models API key | .env file |
| `urls_file` | CSV file with URLs to analyze | config.py |
| `output_dir` | Output directory for results | config.py |

## 🤖 AI Integration

PainPoint.er Scraper supports two AI providers for analysis:

### 1. OpenAI

To use OpenAI's models (default: gpt-3.5-turbo):

1. Add your OpenAI API key to the `.env` file:

```
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

2. When prompted, select option 1 for OpenAI.

### 2. Azure/GitHub Models API

To use Azure's AI models via the GitHub Models API:

1. Add your Azure/GitHub API key to the `.env` file:

```
# Azure/GitHub Models API Key
AZURE_GITHUB_API_KEY=your_azure_github_api_key_here
```

2. When prompted, select option 2 for Azure.

The application will:
1. Prompt you to choose between available AI providers
2. Verify that the required API key is present in the `.env` file
3. Initialize the selected AI client for analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📂 Project Structure

PainPoint.er Scraper has a modular architecture for maintainability and extensibility:

```
├── main.py                 # Main entry point script
├── config.py               # Configuration loading and validation
├── utils.py                # Utility functions (file I/O, URL parsing)
├── scraper.py              # Reddit scraping functionality
├── ai_clients.py           # AI provider client implementations
├── analysis.py             # Text analysis and processing
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
└── urls.csv                # Input file with URLs to analyze
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🔍 Why PainPoint.er Scraper?

In today's competitive software landscape, understanding user pain points is crucial for building successful products. PainPoint.er Scraper automates the process of discovering what users are struggling with and where opportunities exist for new or improved software solutions.

By analyzing real user feedback across multiple platforms, you can:

- **Validate your product ideas** with real-world data
- **Discover underserved markets** and niches
- **Prioritize features** based on user demand
- **Understand competitor weaknesses** to exploit
- **Generate new product ideas** backed by evidence

Whether you're a solo entrepreneur, product manager, or part of a development team, PainPoint.er Scraper helps you make data-driven decisions about what to build next.