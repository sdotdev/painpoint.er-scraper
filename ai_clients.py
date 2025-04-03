# ai_clients.py
import abc
import json
import time # Optional: for adding delays between chunk requests

# --- Abstract Base Class ---
class AIClient(abc.ABC):
    """Abstract base class for AI clients."""
    def __init__(self, api_key):
        if not api_key:
             raise ValueError(f"{self.__class__.__name__} requires an API key.")
        self.api_key = api_key

    @abc.abstractmethod
    def analyze(self, text_chunk_to_analyze, system_prompt):
        """
        Sends a *chunk* of text to the AI for analysis based on the system prompt.

        Args:
            text_chunk_to_analyze (str): A single chunk of text data (e.g., joined comments).
            system_prompt (str): The instruction/role for the AI.

        Returns:
            str or dict: The analysis result for the chunk (string for OpenAI, dict for Azure JSON).
                         Should return dict with 'error' key on failure if JSON expected.

        Raises:
            Exception: If a critical, unrecoverable API call failure occurs (e.g., init failure).
                       May return error dict for recoverable API call errors during chunk processing.
        """
        raise NotImplementedError

# --- OpenAI Implementation ---
try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI library not found. pip install openai")
    OpenAI = None

class OpenAIClient(AIClient):
    """Client for interacting with OpenAI API. Returns string analysis."""
    def __init__(self, api_key):
        super().__init__(api_key)
        if not OpenAI:
             raise ImportError("OpenAI library is required but not installed.")
        try:
            # Consider adding timeout and retry logic for production robustness
            self.client = OpenAI(api_key=self.api_key, timeout=60.0)
            print("OpenAI client initialized.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise

    def analyze(self, text_chunk_to_analyze, system_prompt):
        """Analyzes a text chunk using OpenAI's ChatCompletion API."""
        print("Sending chunk to OpenAI for analysis...")
        # time.sleep(1) # Optional delay
        try:
            # Use a model compatible with your needs and token limits
            # gpt-3.5-turbo is often cost-effective for summarization/analysis
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_chunk_to_analyze}
                ],
                temperature=0.5,
                max_tokens=2000 # Max *output* tokens per chunk
            )
            if response.choices:
                result = response.choices[0].message.content.strip()
                print("Received analysis for chunk from OpenAI.")
                return result
            else:
                print("Warning: OpenAI response had no choices.")
                return "[OpenAI analysis error: No choices in response]"

        except Exception as e:
            print(f"Error during OpenAI API call for chunk: {e}")
            # Let analysis.py handle logging this failure, return an error string
            return f"[OpenAI analysis error for chunk: {e}]"
            # Or raise if you prefer the whole process to stop on one chunk failure
            # raise RuntimeError(f"OpenAI chunk analysis failed: {e}") from e

# --- Azure/GitHub Models Implementation ---
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import JsonSchemaFormat
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    print("Warning: Azure AI Inference libraries not found. pip install azure-ai-inference azure-core")
    ChatCompletionsClient = None
    JsonSchemaFormat = None
    AzureKeyCredential = None

# --- Define the Schema ---
# Ensure this matches the expected structure for merging in analysis.py
PRODUCT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "product_opportunities": {
            "type": "array",
            "description": "List of products users mention using but are unhappy with or seeking alternatives for.",
            "items": {
                "type": "object",
                "properties": {
                    "product_mentioned": {"type": "string", "description": "Name of the product mentioned."},
                    "reason_for_alternative": {"type": "string", "description": "The core reason users are unhappy or want an alternative."},
                    "potential_alternative_idea": {"type": "string", "description": "Brief idea for an alternative product or a key differentiating feature suggested by the comments."}
                },
                "required": ["product_mentioned", "reason_for_alternative"]
            }
        },
        "feature_requests": {
            "type": "array",
            "description": "List of specific features requested for existing products or general software types.",
            "items": {
                "type": "object",
                "properties": {
                    "existing_product": {"type": "string", "description": "The product/software type the feature is requested for (or 'General' if unspecified)."},
                    "requested_feature": {"type": "string", "description": "The specific feature users want."},
                    "user_need": {"type": "string", "description": "The underlying need, goal, or pain point the feature aims to address."}
                },
                "required": ["existing_product", "requested_feature"]
            }
        },
        "pain_points": {
            "type": "array",
            "description": "General pain points mentioned regarding existing software, tools, or workflows.",
            "items": {
                "type": "object",
                "properties": {
                    "area": {"type": "string", "description": "The software category, tool type, or workflow area mentioned (e.g., 'Project Management', 'Note Taking', 'Team Communication')."},
                    "description": {"type": "string", "description": "Description of the pain point or frustration."},
                    "possible_solution_hint": {"type": "string", "description": "Any hints from the comments about what a solution might look like or require."}
                },
                "required": ["area", "description"]
            }
        },
        "new_tool_suggestions": {
            "type": "array",
            "description": "Explicit or implicit suggestions for entirely new tools, software categories, or functionalities not currently available.",
            "items": {
                "type": "object",
                "properties": {
                    "suggestion": {"type": "string", "description": "The core idea for the new tool or functionality."},
                    "justification": {"type": "string", "description": "Why the user thinks this tool is needed or the problem it solves based on the comments."}
                },
                "required": ["suggestion"]
            }
        }
    },
    "required": ["product_opportunities", "feature_requests", "pain_points", "new_tool_suggestions"]
}
# --- End Schema Definition ---


class AzureGitHubClient(AIClient):
    """
    Client for Azure AI Inference using ChatCompletionsClient, requesting structured JSON output.
    Processes one chunk at a time. Uses GitHub token via AzureKeyCredential.
    """
    def __init__(self, api_key): # api_key here is the GitHub Token
        super().__init__(api_key)
        if not ChatCompletionsClient or not AzureKeyCredential or not JsonSchemaFormat:
             raise ImportError("Azure AI Inference libraries (ChatCompletionsClient, JsonSchemaFormat, AzureKeyCredential) are required but not installed/found.")

        # Endpoint for Azure AI Inference Models - adjust if you have a dedicated resource
        self.endpoint = "https://models.inference.ai.azure.com"
        # Model Name - IMPORTANT: Verify the exact name required by the endpoint.
        # Use the specific name provided in the prompt "deepseek-ai/deepseek-v3"
        # self.model_name = "deepseek-ai/deepseek-v3"
        self.model_name = "DeepSeek-V3" # Keep the one that worked, ensure consistency

        print(f"Initializing Azure ChatCompletionsClient for model '{self.model_name}' at endpoint '{self.endpoint}'...")
        try:
            # Timeout can be useful for long-running inference calls
            self.client = ChatCompletionsClient(
                 endpoint=self.endpoint,
                 credential=AzureKeyCredential(self.api_key), # Use the token here
                 model="DeepSeek-V3"
                 # read_timeout=300 # Example: 5 minutes timeout
            )
            print(f"Azure AI ChatCompletionsClient initialized.")
            # It's good practice to make a small test call or check status if possible,
            # but the SDK might not offer a simple 'ping'. Initialization success is a good sign.

        except Exception as e:
            print(f"Error initializing Azure AI ChatCompletionsClient: {e}")
            if "authentication" in str(e).lower():
                 print("Authentication failed. Ensure AZURE_GITHUB_API_KEY in .env is a valid token accepted by the Azure endpoint.")
            elif "endpoint" in str(e).lower():
                 print(f"Endpoint error. Ensure '{self.endpoint}' is the correct Azure AI Inference endpoint URL.")
            # Re-raise critical init errors
            raise

    def analyze(self, text_chunk_to_analyze, system_prompt):
        """Analyzes a text chunk using Azure AI, requesting JSON output via schema."""
        print(f"Sending chunk to Azure AI ({self.model_name}) for JSON analysis...")
        # time.sleep(1.5) # Optional delay

        # --- System Prompt Modification for JSON ---
        json_instruction = (
            "\n\nFormat your response strictly as a JSON object matching the following schema. "
            "Do NOT include any introductory text, explanations, or markdown formatting before or after the JSON object. "
            "Analyze ONLY the text provided in this user message to populate the JSON fields."
            f"\n```json\n{json.dumps(PRODUCT_ANALYSIS_SCHEMA, indent=2)}\n```"
        )
        effective_system_prompt = system_prompt + json_instruction

        try:
            messages = [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": text_chunk_to_analyze}
            ]

            response = self.client.complete(
                messages=messages,
                # model=self.model_name, # Already specified during client initialization
                response_format=JsonSchemaFormat(schema=PRODUCT_ANALYSIS_SCHEMA), # Request JSON output matching schema
                max_tokens=3500,  # Set output token limit
                temperature=0.1,  # Lower temperature for better JSON adherence
                timeout=120.0     # Add timeout to prevent indefinite hanging (2 minutes)
            )

            # --- JSON Parsing and Error Handling for Merging ---
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                # Check finish reason (optional but useful)
                finish_reason = response.choices[0].finish_reason
                if finish_reason != 'stop' and finish_reason != 'tool_calls': # 'tool_calls' might be relevant if using tools
                    print(f"Warning: Azure AI response finish reason was '{finish_reason}' (may indicate truncation or other issues).")

                if message and message.content:
                    content = message.content.strip()
                    # Attempt to find JSON block even if there's slight extraneous text (more robust)
                    json_start = content.find('{')
                    json_end = content.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = content[json_start : json_end + 1]
                        try:
                            parsed_json = json.loads(json_str)
                            # Optional: Validate parsed_json against schema structure here if needed
                            print(f"Received and parsed JSON analysis for chunk from Azure AI ({self.model_name}).")
                            return parsed_json # Success: return the dictionary
                        except json.JSONDecodeError as json_err:
                            print(f"Error: Failed to decode JSON response for chunk: {json_err}")
                            print(f"Attempted to parse:\n{json_str}")
                            print(f"Original content was:\n{content}")
                            return {"error": f"Failed to parse JSON response: {json_err}", "raw_content": content}
                    else:
                         print("Warning: Response content for chunk doesn't appear to contain a valid JSON object.")
                         print(f"Received content:\n{content}")
                         return {"error": "Response was not a JSON object", "raw_content": content}
                else:
                    print("Warning: Azure AI response choice for chunk missing message content.")
                    return {"error": "Response choice missing message content"}
            else:
                # Log usage information if available, even on empty choices
                usage_info = response.usage
                print(f"Warning: Azure AI response for chunk did not contain choices. Usage: {usage_info}")
                return {"error": "No choices returned in response", "usage": usage_info}

        except Exception as e:
            # Catch broader errors during the API call itself
            error_message = f"Error during Azure AI API call for chunk ({self.model_name}): {e}"
            print(error_message)
            # Add specific error type checks if needed (e.g., RateLimitError, APIError from Azure SDK)
            if "quota" in str(e).lower(): print("Quota limit likely reached.")
            elif "model not found" in str(e).lower(): print(f"Model '{self.model_name}' not found or unavailable.")
            elif "schema" in str(e).lower() or "json" in str(e).lower(): print(f"Error likely related to JSON schema/format request. Check model compatibility and schema.")

            # Return an error dictionary for the merging logic
            return {"error": f"Azure AI call failed for chunk: {e}"}


# --- Factory Function ---
def get_ai_client(provider_name, config):
    """Factory function to get an instance of the chosen AI client."""
    provider_name = provider_name.lower()
    if provider_name == "openai":
        api_key = config.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key (OPENAI_API_KEY) not found in config.")
        return OpenAIClient(api_key=api_key)
    elif provider_name == "azure":
        api_key = config.get("azure_github_api_key") # This is the GitHub token
        if not api_key:
            raise ValueError("Azure/GitHub API key (AZURE_GITHUB_API_KEY) not found in config.")
        return AzureGitHubClient(api_key=api_key)
    # Add more providers here as needed
    # elif provider_name == "anthropic":
    #    return AnthropicClient(api_key=config.get("anthropic_api_key"))
    else:
        raise ValueError(f"Unsupported AI provider: {provider_name}. Supported: openai, azure")
