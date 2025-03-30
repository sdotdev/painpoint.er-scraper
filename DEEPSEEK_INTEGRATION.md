# DeepSeek-V3 Integration Guide

This guide explains how to use the DeepSeek-V3 model integration with PainPoint.er Search.

## Prerequisites

1. You need to install the required Azure AI packages:

```bash
pip install azure-ai-inference azure-core
```

2. You need a GitHub token with access to GitHub Models. Set this as an environment variable:

```bash
# On Windows
set GITHUB_TOKEN=your_github_token_here

# On Linux/macOS
export GITHUB_TOKEN=your_github_token_here
```

Alternatively, you can add it to your `.env` file:

```
GITHUB_TOKEN=your_github_token_here
```

## Using DeepSeek-V3

You can use DeepSeek-V3 by specifying it as the API type when running PainPoint.er Search:

```bash
python main.py urls.csv -a deepseek
```

Or interactively select "deepseek" when prompted for the API type.

## How It Works

The DeepSeek-V3 integration uses the Azure AI Inference client to connect to GitHub Models' API. The implementation:

1. Initializes the ModelClient with your GitHub token
2. Formats requests to match DeepSeek-V3's expected input format
3. Processes responses to extract the generated content
4. Handles errors and provides appropriate feedback

The integration is structured to work seamlessly alongside the existing Gemini, OpenAI, and Anthropic integrations.

## Troubleshooting

If you encounter issues:

1. Ensure your GitHub token has access to GitHub Models
2. Check that the required packages are installed
3. Verify your token is correctly set as an environment variable
4. Look for error messages in the application logs

For more information about GitHub Models and DeepSeek-V3, visit the [GitHub Models documentation](https://github.com/features/models).