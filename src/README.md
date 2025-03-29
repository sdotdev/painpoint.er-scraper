# PainPoint.er Search - Source Code

This directory contains the modular components of the PainPoint.er Search tool.

## Modules

- `__init__.py` - Package initialization file
- `url_processor.py` - Handles processing and normalizing URLs from CSV files
- `scrapers.py` - Contains scraper classes for different platforms
- `data_analyzer.py` - Analyzes scraped data to identify pain points and opportunities
- `output_manager.py` - Manages saving analysis results to CSV and JSON files
- `painpointer.py` - Main class that orchestrates the entire process

## Usage

These modules are imported and used by the main script in the root directory. You should not need to interact with these modules directly.

To run the PainPoint.er Search tool, use the main script in the root directory:

```bash
python main.py your_urls.csv
```

See the main README.md in the root directory for more information on usage and options.