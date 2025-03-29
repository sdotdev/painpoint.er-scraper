# PainPoint.er Search 🔍

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Discover Software Pain Points & Product Opportunities

PainPoint.er Search is a powerful, open-source tool that helps entrepreneurs, product managers, and developers identify software pain points and product opportunities by analyzing user feedback from various online communities.

### 🚀 Key Features

- **Multi-Platform Analysis**: Scrape and analyze user feedback from Reddit, Twitter, YouTube, ProductHunt, and more
- **Pain Point Identification**: Automatically detect user complaints and frustrations with existing software
- **Opportunity Scoring**: Calculate demand metrics to prioritize the most promising product ideas
- **AI-Powered Insights**: Leverage Google's Gemini AI (with support for OpenAI and Anthropic) for deeper analysis
- **Comprehensive Reporting**: Generate detailed CSV and JSON reports with actionable insights

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
git clone https://github.com/yourusername/painpointer-search.git
cd painpointer-search
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

Run a basic analysis with default settings:

```bash
python main.py your_urls.csv
```

This will:
1. Process URLs from `your_urls.csv`
2. Scrape data from the last 7 days
3. Analyze for software mentions and pain points
4. Generate product ideas based on identified opportunities
5. Save results to the `./output` directory

## 💡 Usage Examples

### Basic Usage

```bash
python main.py urls.csv
```

### Specify Lookback Period

```bash
python main.py urls.csv --days 30
```

### Custom Output Directory

```bash
python main.py urls.csv --output ./my_results
```

### With AI Analysis (Google Gemini)

```bash
python main.py urls.csv --api-key YOUR_GEMINI_API_KEY
```

### With Alternative AI Provider

```bash
python main.py urls.csv --api-key YOUR_API_KEY --api-type openai
```

## 📄 Input Format

The input CSV file must contain a column named `url` with the URLs to analyze. Additional columns are preserved and included in the analysis results.

Example `urls.csv`:

```
url,category,notes
https://reddit.com/r/productivity,productivity,Popular productivity subreddit
twitter.com/search?q=notion%20issues,note-taking,Search for Notion issues
youtube.com/watch?v=example_video_id,video_editing,Video with many comments
```

## 📊 Output Format

PainPoint.er Search generates several output files:

1. `painpointer_results.csv` - Main results with product ideas and demand scores
2. `painpointer_results_software.csv` - Detailed software mentions
3. `painpointer_results_pain_points.csv` - Identified pain points
4. `painpointer_results_demand.csv` - Demand metrics by platform
5. `painpointer_results_ideas.csv` - Generated product ideas
6. `painpointer_results.json` - Complete results in JSON format
7. `raw_data.json` - Raw scraped data

## ⚙️ Advanced Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|--------|
| `csv_file` | CSV file with URLs to analyze | urls.csv |
| `-d, --days` | Number of days to look back | 7 |
| `-o, --output` | Output directory for results | ./output |
| `-k, --api-key` | API key for AI analysis | (Optional) |
| `-a, --api-type` | AI API to use (gemini, openai, anthropic, none) | gemini |

## 🤖 AI Integration

PainPoint.er Search supports multiple AI providers for enhanced analysis. You can provide API keys in two ways:

### Using Environment Variables (Recommended)

Create a `.env` file in the project root with your API keys:

```
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

The application will automatically:
1. Detect available API keys in your `.env` file
2. Set the default AI provider based on available keys
3. Use the appropriate key for analysis

You can still override these defaults using command-line arguments.

### Using Command Line Arguments

#### Google Gemini

```bash
python main.py urls.csv --api-key YOUR_GEMINI_API_KEY --api-type gemini
```

#### OpenAI

```bash
python main.py urls.csv --api-key YOUR_OPENAI_API_KEY --api-type openai
```

#### Anthropic Claude

```bash
python main.py urls.csv --api-key YOUR_ANTHROPIC_API_KEY --api-type anthropic
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📂 Project Structure

PainPoint.er Search has been modularized for better maintainability and extensibility:

```
├── main.py                 # Main entry point script
├── requirements.txt        # Python dependencies
├── sample_urls.csv         # Example input file
└── src/                    # Source code modules
    ├── painpointer.py      # Core PainPointerSearch class
    ├── url_processor.py    # URL processing and normalization
    ├── scrapers.py         # Platform-specific web scrapers
    ├── data_analyzer.py    # Analysis and AI integration
    └── output_manager.py   # Results formatting and export
```

### Module Descriptions

- **main.py**: Entry point that handles command-line arguments and runs the search process
- **src/painpointer.py**: Core class that orchestrates the entire search and analysis workflow
- **src/url_processor.py**: Handles CSV parsing and URL normalization
- **src/scrapers.py**: Contains platform-specific scrapers for Reddit, Twitter, etc.
- **src/data_analyzer.py**: Analyzes scraped data and integrates with AI APIs
- **src/output_manager.py**: Formats and exports results to various file formats

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🔍 Why PainPoint.er Search?

In today's competitive software landscape, understanding user pain points is crucial for building successful products. PainPoint.er Search automates the process of discovering what users are struggling with and where opportunities exist for new or improved software solutions.

By analyzing real user feedback across multiple platforms, you can:

- **Validate your product ideas** with real-world data
- **Discover underserved markets** and niches
- **Prioritize features** based on user demand
- **Understand competitor weaknesses** to exploit
- **Generate new product ideas** backed by evidence

Whether you're a solo entrepreneur, product manager, or part of a development team, PainPoint.er Search helps you make data-driven decisions about what to build next.