import json
import re
import datetime
import statistics
import collections
import math
import difflib
import logging
from typing import Dict, List, Optional, Tuple, Counter, Any, Set

# Optional dependency handling
try:
    import spacy
    try:
        NLP = spacy.load("en_core_web_sm")
        logging.info("spaCy model 'en_core_web_sm' loaded successfully.")
    except OSError:
        logging.warning("spaCy model 'en_core_web_sm' not found. "
                        "Run 'python -m spacy download en_core_web_sm' for better software name extraction. "
                        "Falling back to regex.")
        NLP = None
except ImportError:
    logging.warning("spaCy library not found. Run 'pip install spacy' and "
                    "'python -m spacy download en_core_web_sm' for improved software name extraction. "
                    "Falling back to regex.")
    NLP = None

# Using TextBlob for basic sentiment, can be swapped if needed
try:
    from textblob import TextBlob
except ImportError:
    logging.error("TextBlob library not found. Please install it: pip install textblob")
    # Provide a dummy class if TextBlob is not available to avoid crashing
    class DummySentiment:
        polarity = 0.0
        subjectivity = 0.0

    class DummyTextBlob:
        def __init__(self, text):
            self.sentiment = DummySentiment()

    TextBlob = DummyTextBlob # type: ignore # Assign dummy class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
SOFTWARE_CONTEXT_KEYWORDS = {
    'app', 'tool', 'software', 'platform', 'service', 'api', 'sdk',
    'library', 'framework', 'website', 'program', 'system', 'crm',
    'erp', 'saas', 'extension', 'plugin', 'bot', 'database', 'editor'
}
POTENTIAL_SOFTWARE_REGEX = re.compile(
    r'\b([A-Z][a-zA-Z0-9]*(?:[.\s-][A-Z][a-zA-Z0-9]*)*' # Multi-word like "Google Cloud" or "VS Code"
    r'(?:\.js|\.ai|\.io|\.app)?'                       # Optional common endings
    r')\b'
)
PAIN_POINT_KEYWORDS = {
    'critical': ['crash', 'bug', 'error', 'broken', 'unusable', 'terrible', 'awful', 'hate', 'fail', 'freeze', 'exploit', 'vulnerability', 'inconsistent'],
    'major': ['slow', 'frustrating', 'annoying', 'expensive', 'complicated', 'difficult', 'confusing', 'lag', 'unreliable', 'poor', 'security', 'privacy'],
    'minor': ['issue', 'problem', 'glitch', 'missing', 'lack', 'needs', 'dislike', 'limit', 'concern', 'tricky', 'weird']
}
FEATURE_REQUEST_KEYWORDS = {
    'request': ['wish', 'hope', 'feature', 'integrate', 'add', 'support', 'api', 'plugin', 'extension', 'capability', 'option', 'more', 'better', 'improve', 'want', 'develop', 'create', 'allow', 'enable']
}
ALL_ISSUE_KEYWORDS = {k: v for d in [PAIN_POINT_KEYWORDS, FEATURE_REQUEST_KEYWORDS] for k, v in d.items()}


class DataAnalyzer:
    """Analyze scraped data to identify pain points, features, and opportunities."""

    def __init__(self, api_key: Optional[str] = None, api_type: str = 'gemini'):
        """
        Initializes the DataAnalyzer.

        Args:
            api_key: Optional API key for AI services (Gemini, OpenAI, Anthropic).
            api_type: The type of AI service to use ('gemini', 'openai', 'anthropic'). Defaults to 'gemini'.
        """
        self.api_key = api_key
        self.api_type = api_type.lower() # Ensure lowercase for comparisons
        self.sentiment_analyzer = TextBlob
        self.genai = None # Stores the configured google.generativeai module (if using configure)
        self.genai_client = None # Stores the google.generativeai.Client instance (if using Client)
        self.openai = None # Stores the OpenAI client module/object
        self.anthropic_client = None # Stores the Anthropic client instance

        # Stores demand metrics calculated during analysis for potential reuse by AI insights
        self.demand_metrics: Dict[str, Any] = {}

        if self.api_key:
            self._setup_ai_connection()
        else:
             logging.info("No API key provided. AI features (software extraction assist, idea generation, insights) will be disabled.")


    def _setup_ai_connection(self):
        """Set up connection to the specified AI API."""
        logging.info(f"Attempting to set up AI connection for: {self.api_type}")

        # --- Gemini Setup ---
        # Handles both genai.Client (newer) and genai.configure (older) initialization methods.
        if self.api_type == 'gemini':
            if not self.api_key:
                logging.warning("Gemini API key missing. Cannot initialize Gemini.")
                return
            try:
                # Dynamically import 'google.generativeai' only when needed
                import google.generativeai as genai
                self.genai = genai # Store module reference

                # Try initializing using the Client API (preferred)
                try:
                    # Check if the Client class exists in the imported module
                    if hasattr(genai, 'GenerativeModel') or hasattr(genai, 'Client'): # Check for common client/model classes
                        # Note: Modern library uses GenerativeModel usually, but check Client for broader compatibility.
                        # Configuration is often done globally via configure, then model instantiated.
                        # Let's prioritize configure as it's simpler and common.
                        genai.configure(api_key=self.api_key)
                        logging.info("Google Gemini AI configured successfully using genai.configure().")
                        # We don't create a client instance here, rely on configure + model instantiation later.
                        # Set genai_client to None to ensure later calls use the configure path.
                        self.genai_client = None

                    else:
                         # Fallback if configure is somehow unavailable but Client exists (less common)
                         # This path is less likely with current library versions.
                         logging.warning("genai.configure() not found or failed, attempting genai.Client() initialization (less common).")
                         if hasattr(genai, 'Client'):
                             self.genai_client = genai.Client(api_key=self.api_key)
                             logging.info('Google Gemini AI client initialized successfully using genai.Client().')
                         else:
                             raise AttributeError("Neither genai.configure nor genai.Client seems available.")

                except Exception as e:
                    logging.error(f"Failed to initialize Google Gemini AI: {e}. "
                                  "Ensure 'google-generativeai' is installed and API key is valid. "
                                  "Check Google Cloud project permissions if applicable.")
                    self.genai = None # Nullify on error
                    self.genai_client = None

            except ImportError:
                logging.error("Google Generative AI SDK ('google-generativeai') not found. "
                              "Please install it: pip install google-generativeai")
                self.genai = None
                self.genai_client = None
            except Exception as e: # Catch any other unexpected errors during import/setup
                 logging.error(f"An unexpected error occurred during Gemini setup: {e}")
                 self.genai = None
                 self.genai_client = None


        # --- OpenAI Setup ---
        elif self.api_type == 'openai':
            if not self.api_key:
                logging.warning("OpenAI API key missing. Cannot initialize OpenAI.")
                return
            try:
                import openai
                # Use the modern client initialization
                self.openai = openai.OpenAI(api_key=self.api_key)
                logging.info('OpenAI client initialized successfully.')
                # Test connection lightly (optional, remove if causes issues)
                # try:
                #     self.openai.models.list()
                #     logging.info("OpenAI connection successful (listed models).")
                # except Exception as conn_err:
                #     logging.warning(f"Could not confirm OpenAI connection via models.list: {conn_err}")

            except ImportError:
                logging.error("OpenAI library ('openai') not found. Please install it: pip install openai")
                self.openai = None
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI: {e}. Check API key and library version.")
                self.openai = None

        # --- Anthropic Setup ---
        elif self.api_type == 'anthropic':
            if not self.api_key:
                logging.warning("Anthropic API key missing. Cannot initialize Anthropic.")
                return
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
                logging.info('Anthropic client initialized successfully.')
                 # Test connection lightly (optional)
                # try:
                #     # A simple operation like creating a small message could test auth
                #     self.anthropic_client.messages.create(max_tokens=1, model="claude-3-haiku-20240307", messages=[{"role":"user", "content":"test"}])
                #     logging.info("Anthropic connection successful (test message created).")
                # except Exception as conn_err:
                #      logging.warning(f"Could not confirm Anthropic connection via test message: {conn_err}")

            except ImportError:
                logging.error("Anthropic library ('anthropic') not found. Please install it: pip install anthropic")
                self.anthropic_client = None
            except Exception as e:
                logging.error(f"Failed to initialize Anthropic: {e}. Check API key.")
                self.anthropic_client = None

        else:
             logging.error(f"Unsupported API type: '{self.api_type}'. Choose 'gemini', 'openai', or 'anthropic'.")


    def analyze_data(self, scraped_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze scraped data to identify pain points, features, opportunities, etc.

        Args:
            scraped_data: A list of dictionaries, where each dictionary represents
                          a scraped data point (e.g., a post, comment). Expected
                          keys might include 'platform', 'title', 'post_content',
                          'comments', 'timestamp', 'url', 'upvotes', etc.

        Returns:
            A dictionary containing various analysis results.
        """
        if not scraped_data:
            logging.warning("No data provided for analysis.")
            return self._empty_analysis_results()

        logging.info(f"Starting analysis on {len(scraped_data)} data points.")

        # --- Core Analysis Steps ---
        all_text = self._combine_text_data(scraped_data)
        logging.info(f"Combined text length: {len(all_text)} characters.")

        software_mentions_detailed = self._extract_software_mentions(all_text, scraped_data)
        logging.info(f"Identified {len(software_mentions_detailed)} distinct software products (meeting threshold).")

        issues = self._extract_pain_points_and_features(software_mentions_detailed)
        pain_points_data = [item for item in issues if item['type'] == 'pain_point']
        feature_requests_data = [item for item in issues if item['type'] == 'feature_request']
        logging.info(f"Extracted {sum(p['pain_point_summary']['count'] for p in pain_points_data)} pain points across {len(pain_points_data)} software items.")
        logging.info(f"Extracted {sum(f['feature_request_summary']['count'] for f in feature_requests_data)} feature requests across {len(feature_requests_data)} software items.")

        # Prepare simplified list for output, but use detailed list internally
        software_names_with_mentions = [{'name': s['name'], 'mentions': s['mentions']} for s in software_mentions_detailed]

        sentiment_analysis = self._analyze_sentiment(software_mentions_detailed)
        logging.info("Completed sentiment analysis.")

        # Calculate demand metrics and store them in the instance variable for potential use in AI insights
        self.demand_metrics = self._calculate_demand_metrics(scraped_data, software_mentions_detailed, sentiment_analysis)
        logging.info("Calculated demand metrics.")

        statistical_analysis = self._perform_statistical_analysis(scraped_data, software_mentions_detailed, issues)
        logging.info("Performed statistical analysis.")

        trend_analysis = self._detect_trends(scraped_data, software_mentions_detailed, issues)
        logging.info("Detected trends.")

        # Generate product ideas using extracted issues and metrics
        product_ideas = self._generate_product_ideas(
            pain_points_data,
            feature_requests_data,
            self.demand_metrics, # Pass the calculated metrics
            statistical_analysis.get('co_occurrence', {})
        )
        logging.info(f"Generated {len(product_ideas)} product ideas (AI + fallback).")

        # --- AI Summary Insights (Optional) ---
        ai_insights = {}
        # Check if any AI client was successfully initialized
        if self.api_key and (self.genai or self.genai_client or self.openai or self.anthropic_client):
            logging.info(f"Generating AI insights summary using {self.api_type}...")
            try:
                ai_insights = self._generate_ai_insights(
                    software_mentions_detailed, # Pass detailed list for better AI context
                    pain_points_data,
                    feature_requests_data,
                    sentiment_analysis,
                    statistical_analysis
                    # self.demand_metrics is accessed via self within _generate_ai_insights
                )
                if 'error' not in ai_insights:
                     logging.info("AI insights generation complete.")
                else:
                     logging.warning(f"AI insights generation failed: {ai_insights.get('error', 'Unknown reason')}")
            except Exception as ai_err:
                 logging.error(f"An unexpected error occurred during AI insights generation: {ai_err}", exc_info=True)
                 ai_insights = {'error': f"Unexpected error during AI insights generation: {str(ai_err)}"}
        else:
            logging.info("Skipping AI insights generation (API key or client not available/initialized).")

        # --- Compile Final Results ---
        return {
            'summary': {
                'total_data_points': len(scraped_data),
                'total_software_identified': len(software_mentions_detailed),
                'total_pain_points': sum(p['pain_point_summary']['count'] for p in pain_points_data),
                'total_feature_requests': sum(f['feature_request_summary']['count'] for f in feature_requests_data),
                'product_ideas_generated': len(product_ideas),
            },
            'software_mentions': software_names_with_mentions, # Simplified list for output
            'pain_points': pain_points_data,
            'feature_requests': feature_requests_data,
            'demand_metrics': self.demand_metrics, # Use the stored metrics
            'sentiment_analysis': sentiment_analysis,
            'statistical_analysis': statistical_analysis,
            'trend_analysis': trend_analysis,
            'product_ideas': product_ideas,
            'ai_insights': ai_insights
        }

    def _empty_analysis_results(self) -> Dict[str, Any]:
        """Returns a dictionary with empty results for all analysis keys."""
        return {
            'summary': {}, 'software_mentions': [], 'pain_points': [], 'feature_requests': [],
            'demand_metrics': {}, 'sentiment_analysis': {}, 'statistical_analysis': {},
            'trend_analysis': {}, 'product_ideas': [], 'ai_insights': {}
        }

    def _combine_text_data(self, scraped_data: List[Dict[str, Any]]) -> str:
        """Combine all relevant text fields from a list of scraped data items."""
        text_parts = []
        for data in scraped_data:
            # Extract text robustly, handling missing keys gracefully
            title = data.get('title', '')
            # Common keys for main content across different platforms
            post_content = data.get('post_content', data.get('description',
                                    data.get('tweet_content', data.get('body', data.get('content', '')))))
            # Common keys for comments/replies
            comments_list = data.get('comments', data.get('replies', []))

            # Append non-empty text parts with labels for clarity
            if title: text_parts.append(f"TITLE: {title}")
            if post_content: text_parts.append(f"CONTENT: {post_content}")

            for i, comment in enumerate(comments_list):
                # Handle cases where comment is a string or a dict
                comment_text = ''
                if isinstance(comment, dict):
                    comment_text = comment.get('text', comment.get('body', ''))
                elif isinstance(comment, str):
                    comment_text = comment

                if comment_text:
                    text_parts.append(f"COMMENT_{i+1}: {comment_text}")

        # Join parts with double newline to ensure separation
        return "\n\n".join(filter(None, text_parts)) # Filter out potential empty strings

    def _extract_software_mentions(self, text: str, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract mentions of software products using AI (if available) or NER/Regex fallback.
        Returns a detailed list including context and mention count, filtered by minimum mentions.
        """
        mentioned_software: Dict[str, Dict[str, Any]] = {}
        ai_identified_names: Set[str] = set()
        max_ai_text_length = 30000 # Limit context for AI prompt to avoid excessive length/cost
        ai_available = bool(self.api_key and (self.genai or self.genai_client or self.openai or self.anthropic_client))

        # 1. Attempt AI Extraction (if available)
        if ai_available:
            try:
                logging.info(f"Attempting software extraction using AI ({self.api_type})...")
                # Truncate text if it exceeds the limit
                text_sample = text[:max_ai_text_length] if len(text) > max_ai_text_length else text
                if len(text) > max_ai_text_length:
                     logging.warning(f"Input text truncated to {max_ai_text_length} chars for AI software extraction.")

                ai_prompt = f"""
                Analyze the following text from online discussions (Reddit, Twitter, YouTube, Product Hunt etc.)
                and identify all specific software products, applications, tools, libraries, frameworks, APIs, or platforms mentioned.
                Be precise. For example, identify "VS Code" or "Google Docs", not just "Google".
                Exclude generic terms like 'software', 'app', 'tool' unless they are part of a specific name (like '.app').
                Consider variations like 'VSCode', 'vs code'.

                For each distinct software identified, provide its canonical name (e.g., "Visual Studio Code" instead of "VS Code" if context allows, otherwise use the mentioned form).

                Format your response STRICTLY as a JSON array of strings, where each string is a software name.
                Example: ["Slack", "Zoom", "Visual Studio Code", "React", "AWS Lambda", "Figma"]

                Text to analyze:
                ---
                {text_sample}
                ---
                Respond only with the JSON array.
                """

                raw_ai_output = None
                model_to_use = 'models/gemini-1.5-flash-latest' # Default Gemini model

                # --- AI Call Logic ---
                if self.api_type == 'gemini' and (self.genai or self.genai_client):
                    # Prioritize configured genai module (uses global configure)
                    if self.genai and hasattr(self.genai, 'GenerativeModel'):
                        logging.info(f"Using Gemini model '{model_to_use}' via configured genai module.")
                        model = self.genai.GenerativeModel(model_to_use)
                        response = model.generate_content(ai_prompt)
                        raw_ai_output = response.text
                    # Fallback to client instance if that was initialized instead (less common now)
                    elif self.genai_client and hasattr(self.genai_client, 'generate_content'):
                        logging.warning(f"Using Gemini model '{model_to_use}' via genai.Client instance (less common).")
                        response = self.genai_client.generate_content(model=model_to_use, contents=ai_prompt) # Adjust call signature if needed
                        raw_ai_output = response.text
                    else:
                        logging.error("Gemini AI object (genai module or client) not configured correctly for content generation.")

                elif self.api_type == 'openai' and self.openai:
                     logging.info("Using OpenAI model 'gpt-4o-mini'.")
                     response = self.openai.chat.completions.create(
                         model='gpt-4o-mini', # Cost-effective and capable model
                         response_format={'type': 'json_object'},
                         messages=[
                             {'role': 'system', 'content': 'You extract software names from text and return a JSON array of strings.'},
                             {'role': 'user', 'content': ai_prompt}
                         ]
                     )
                     raw_ai_output = response.choices[0].message.content

                elif self.api_type == 'anthropic' and self.anthropic_client:
                     logging.info("Using Anthropic model 'claude-3-haiku-20240307'.")
                     response = self.anthropic_client.messages.create(
                         model='claude-3-haiku-20240307', # Fast and cost-effective model
                         max_tokens=1000,
                         messages=[{'role': 'user', 'content': ai_prompt}]
                     )
                     raw_ai_output = response.content[0].text
                # --- End AI Call Logic ---

                if raw_ai_output:
                    # Extract JSON array from the response (handle potential markdown/text wrapping)
                    json_match = re.search(r'\[.*?\]', raw_ai_output, re.DOTALL)
                    if json_match:
                        try:
                            software_list = json.loads(json_match.group(0))
                            if isinstance(software_list, list) and all(isinstance(item, str) for item in software_list):
                                ai_identified_names.update(name.strip() for name in software_list if name.strip() and len(name.strip()) > 1) # Add basic length filter
                                logging.info(f"AI identified {len(ai_identified_names)} potential software names.")
                            else:
                                logging.warning(f"AI response JSON was not a list of strings: {software_list}")
                        except json.JSONDecodeError as e:
                            logging.warning(f"AI response was not valid JSON: {e}. Raw output starts with: {raw_ai_output[:200]}...")
                    else:
                        logging.warning(f"Could not find JSON array '[]' in AI response. Raw output starts with: {raw_ai_output[:200]}...")
                else:
                    logging.warning(f"AI ({self.api_type}) returned no output for software extraction.")

            except Exception as e:
                logging.error(f"Error during AI software extraction ({self.api_type}): {e}", exc_info=True)
                # Continue with fallback methods even if AI fails

        # 2. Fallback/Augmentation: Use NER and Regex
        logging.info("Running NER/Regex fallback/augmentation for software extraction...")
        fallback_identified_names: Counter[str] = collections.Counter()
        context_map: Dict[str, List[str]] = collections.defaultdict(list)
        processed_text_fallback = text # Use full text for fallback

        # Use spaCy NER if available
        if NLP:
            try:
                # Process in chunks if text is very long to avoid memory issues
                max_spacy_len = 1_000_000
                if len(processed_text_fallback) > max_spacy_len:
                     logging.warning(f"Text length > {max_spacy_len}, processing only the start for spaCy NER.")
                     processed_text_fallback = processed_text_fallback[:max_spacy_len]

                doc = NLP(processed_text_fallback)
                possible_entities = [ent.text.strip() for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG']]
                logging.info(f"spaCy NER found {len(possible_entities)} potential PRODUCT/ORG entities.")

                # Filter entities based on capitalization, length, and context
                for entity in set(possible_entities): # Use set to avoid redundant checks for same entity text
                    if len(entity) > 1 and (entity[0].isupper() or any(c.isdigit() for c in entity)):
                        # Find matches and check context keywords
                        try:
                            # Use regex finditer for context checking
                            pattern = r'\b' + re.escape(entity) + r'\b'
                            for match in re.finditer(pattern, processed_text_fallback, re.IGNORECASE):
                                start = max(0, match.start() - 75) # Slightly wider context window
                                end = min(len(processed_text_fallback), match.end() + 75)
                                context_snippet = processed_text_fallback[start:end].lower()
                                if any(keyword in context_snippet for keyword in SOFTWARE_CONTEXT_KEYWORDS):
                                    normalized_name = entity.rstrip('.') # Basic normalization
                                    fallback_identified_names[normalized_name] += 1
                                    # Store context only if needed for debugging, might remove later
                                    # if len(context_map[normalized_name]) < 5:
                                    #    context_map[normalized_name].append(processed_text_fallback[start:end].replace('\n', ' ').strip())
                                    break # Count once per unique entity text found in relevant context
                        except re.error:
                            logging.warning(f"Regex error processing spaCy entity: {entity}")
                            continue # Skip this entity if regex fails
            except Exception as e:
                logging.error(f"Error during spaCy NER processing: {e}", exc_info=True)

        # Use Regex as a further fallback or supplement
        try:
            for match in POTENTIAL_SOFTWARE_REGEX.finditer(text): # Use original full text for regex
                potential_name = match.group(1).strip().rstrip('.')
                # Basic filtering: length > 1, not purely numeric, avoid common acronyms/words likely not software
                common_non_software = {'API', 'SDK', 'AI', 'ML', 'UI', 'UX', 'IT', 'CEO', 'CTO', 'PM'}
                if len(potential_name) > 1 and \
                   not potential_name.isdigit() and \
                   potential_name.upper() not in common_non_software and \
                   potential_name.lower() not in SOFTWARE_CONTEXT_KEYWORDS: # Avoid matching keywords themselves
                    # Check context around the regex match
                    start = max(0, match.start() - 75)
                    end = min(len(text), match.end() + 75)
                    context_snippet = text[start:end].lower()
                    # Require context keyword OR check if name ends with common software suffix
                    has_context_keyword = any(keyword in context_snippet for keyword in SOFTWARE_CONTEXT_KEYWORDS)
                    has_suffix = any(potential_name.lower().endswith(suffix) for suffix in ['.js', '.ai', '.io', '.app', 'db'])

                    if has_context_keyword or has_suffix:
                        fallback_identified_names[potential_name] += 1
                        # Store context only if needed
                        # if len(context_map[potential_name]) < 5:
                        #    context_map[potential_name].append(text[start:end].replace('\n', ' ').strip())
        except Exception as e:
            logging.error(f"Error during Regex software extraction: {e}", exc_info=True)

        logging.info(f"Fallback methods identified {len(fallback_identified_names)} potential names with counts.")

        # 3. Consolidate Results & Recalculate Mentions with Context
        # Combine candidates from AI and Fallback methods
        final_candidates = ai_identified_names.union(fallback_identified_names.keys())
        logging.info(f"Consolidating {len(final_candidates)} unique candidate names.")

        # Recalculate mentions and gather context accurately across the entire structured dataset
        final_mentions_data: Dict[str, Dict[str, Any]] = collections.defaultdict(
            lambda: {'name': '', 'mentions': 0, 'context': []}
        )

        # Iterate through the original structured data for accurate counting and context association
        for data_item in scraped_data:
            item_text = self._combine_text_data([data_item]) # Get text for this specific item
            if not item_text: continue # Skip if item has no text content

            item_url = data_item.get('url', 'N/A')
            timestamp = data_item.get('timestamp', 'N/A') # Get timestamp if available

            for name in final_candidates:
                try:
                    # Use word boundaries and case-insensitive matching
                    escaped_name = re.escape(name)
                    pattern = re.compile(r'\b' + escaped_name + r'\b', re.IGNORECASE)
                    matches = list(pattern.finditer(item_text))

                    if matches:
                        mention_count_in_item = len(matches)
                        final_mentions_data[name]['name'] = name # Ensure name is set
                        final_mentions_data[name]['mentions'] += mention_count_in_item

                        # Add context from the first match in this item, limit total context stored
                        if len(final_mentions_data[name]['context']) < 15: # Store max 15 context snippets per software
                            match_obj = matches[0]
                            start = max(0, match_obj.start() - 100) # Wider context snippet
                            end = min(len(item_text), match_obj.end() + 100)
                            context_str = item_text[start:end].replace('\n', ' ').strip()
                            # Include source URL and timestamp in context string
                            context_with_meta = f"{context_str} (Source: {item_url}, Timestamp: {timestamp})"
                            final_mentions_data[name]['context'].append(context_with_meta)

                except re.error:
                    logging.warning(f"Regex error searching for software name: {name} in item: {item_url}")
                    continue # Skip this name for this item if regex fails
                except Exception as e:
                    logging.error(f"Unexpected error processing name '{name}' in item '{item_url}': {e}", exc_info=False)
                    continue


        # Filter out low-mention candidates (likely noise or irrelevant mentions)
        min_mentions_threshold = 2 # Require at least 2 mentions across the entire dataset
        filtered_mentions_list = [
            details for details in final_mentions_data.values()
            if details['mentions'] >= min_mentions_threshold
        ]

        # Sort by total mention count descending
        return sorted(filtered_mentions_list, key=lambda x: x['mentions'], reverse=True)


    def _extract_pain_points_and_features(self, software_mentions_detailed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts pain points and feature requests related to software mentions using keyword matching and sentiment analysis.

        Args:
            software_mentions_detailed: The detailed list of software mentions, including context snippets.

        Returns:
            A list of dictionaries, each representing a software and its associated issues (pain points/feature requests).
            Issues are categorized by type and severity.
        """
        all_issues_data = []
        # Create a flat map of keyword -> category for efficient lookup
        issue_keywords_flat: Dict[str, str] = {
            kw.lower(): cat
            for cat, kws in ALL_ISSUE_KEYWORDS.items() for kw in kws
        }
        # Define severity order for pain points
        pain_severity_order = ['critical', 'major', 'minor']

        logging.info("Extracting pain points and feature requests from software mention contexts...")
        for software in software_mentions_detailed:
            software_name = software['name']
            contexts = software['context'] # Contexts now include source URL and timestamp
            if not contexts:
                continue

            software_issues: List[Dict[str, Any]] = []

            for context_with_meta in contexts:
                # Extract the actual text content from the metadata string if needed
                # Assuming context_with_meta format: "Text... (Source: URL, Timestamp: TIME)"
                context_match = re.match(r"^(.*)\s+\(Source:.*,\s+Timestamp:.*\)$", context_with_meta, re.DOTALL)
                if context_match:
                    context_text = context_match.group(1).strip()
                else:
                    context_text = context_with_meta # Use the whole string if parsing fails

                context_lower = context_text.lower()
                found_keywords = []
                matched_categories: Set[str] = set()

                # Find all matching keywords in the context
                # Use word boundaries to avoid matching parts of words
                for keyword, category in issue_keywords_flat.items():
                    try:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', context_lower):
                            found_keywords.append(keyword)
                            matched_categories.add(category)
                    except re.error:
                         logging.warning(f"Regex error searching for keyword: {keyword}")
                         continue # Skip keyword if regex fails

                if not found_keywords:
                    continue # Skip context if no relevant keywords found

                # Determine the primary category and severity
                best_category = None
                severity = None
                is_request = 'request' in matched_categories
                pain_categories_found = {cat for cat in matched_categories if cat in pain_severity_order}

                if pain_categories_found:
                    # Prioritize the most severe pain point category found
                    current_severity_level = -1
                    for i, sev in enumerate(pain_severity_order):
                         if sev in pain_categories_found:
                              if current_severity_level == -1 or i < current_severity_level:
                                   current_severity_level = i
                                   severity = sev
                                   best_category = sev # Assign the pain category

                elif is_request: # If only request keywords found
                    best_category = 'request'
                    severity = None # Severity not applicable to requests

                # If a category was determined, proceed to sentiment analysis and finalize issue type
                if best_category:
                    try:
                         # Analyze sentiment of the original context text (before lowercasing)
                         sentiment_score = self.sentiment_analyzer(context_text).sentiment.polarity
                    except Exception as e:
                         logging.warning(f"Sentiment analysis failed for context: {context_text[:100]}... Error: {e}")
                         sentiment_score = 0.0 # Default to neutral if analysis fails

                    issue_type = 'unknown'
                    if best_category == 'request':
                        # Consider it a feature request unless sentiment is very negative
                        issue_type = 'feature_request' if sentiment_score > -0.3 else 'pain_point'
                        if issue_type == 'pain_point': severity = 'minor' # If negative sentiment overrides request keyword

                    elif severity: # If a pain point category was identified
                         issue_type = 'pain_point'
                         # Optionally escalate severity based on strong negative sentiment
                         if sentiment_score < -0.6:
                              if severity == 'major': severity = 'critical'
                              elif severity == 'minor': severity = 'major'

                    # Add the identified issue
                    software_issues.append({
                        'context': context_with_meta, # Store context with metadata
                        'keywords': found_keywords,
                        'type': issue_type, # 'pain_point' or 'feature_request'
                        'severity': severity, # 'critical', 'major', 'minor', or None
                        'sentiment_score': round(sentiment_score, 3)
                    })

            # After processing all contexts for a software, aggregate the results
            if software_issues:
                pain_points = [iss for iss in software_issues if iss['type'] == 'pain_point']
                feature_requests = [iss for iss in software_issues if iss['type'] == 'feature_request']

                pain_severity_counts = collections.Counter(pp['severity'] for pp in pain_points if pp['severity'])

                # Determine overall type based on which category has more issues
                primary_type = 'pain_point' if len(pain_points) >= len(feature_requests) else 'feature_request'

                all_issues_data.append({
                    'software': software_name,
                    'issues': software_issues, # Keep all detailed issues
                    'type': primary_type, # Primary issue type for this software
                    'pain_point_summary': {
                        'count': len(pain_points),
                        'severity_counts': dict(pain_severity_counts) # Convert Counter to dict for JSON compatibility
                    },
                    'feature_request_summary': {
                        'count': len(feature_requests)
                    },
                    'total_issues': len(software_issues)
                })

        # Sort results: Prioritize software with critical pain points, then by total issue count
        return sorted(all_issues_data, key=lambda x: (
            x['pain_point_summary']['severity_counts'].get('critical', 0), # Sort critical count descending
            x['total_issues'] # Then by total issues descending
        ), reverse=True)


    def _calculate_demand_metrics(self, scraped_data: List[Dict[str, Any]], software_mentions_detailed: List[Dict[str, Any]], sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive demand metrics for mentioned software."""
        metrics: Dict[str, Any] = {}
        total_data_points = len(scraped_data)
        if total_data_points == 0:
            logging.warning("Cannot calculate demand metrics: No scraped data provided.")
            return {}
        if not software_mentions_detailed:
            logging.warning("Cannot calculate demand metrics: No software mentions found.")
            return {}

        logging.info("Calculating demand metrics...")
        # Calculate total mentions across all identified software AFTER filtering
        total_all_mentions = sum(s['mentions'] for s in software_mentions_detailed)
        if total_all_mentions == 0: total_all_mentions = 1 # Avoid division by zero

        # Prepare timestamp conversion utility
        def parse_timestamp(ts_str: Optional[str]) -> Optional[datetime.datetime]:
            if not ts_str: return None
            try:
                # Handle ISO format with or without 'Z', microseconds optional
                ts_str = ts_str.replace('Z', '+00:00')
                # Try parsing with microseconds
                try:
                     return datetime.datetime.fromisoformat(ts_str).astimezone(datetime.timezone.utc)
                except ValueError:
                     # Try parsing without microseconds if the first attempt failed
                     return datetime.datetime.fromisoformat(ts_str.split('.')[0] + ts_str.split('.')[-1][6:] if '.' in ts_str else ts_str).astimezone(datetime.timezone.utc)

            except (ValueError, TypeError, IndexError) as e:
                # logging.debug(f"Could not parse timestamp '{ts_str}': {e}") # Make debug to reduce noise
                return None

        # Pre-process data for faster lookups: Map data by software mention
        data_by_software: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        all_software_names = {sw['name'] for sw in software_mentions_detailed}

        for data_item in scraped_data:
            item_text = self._combine_text_data([data_item])
            if not item_text: continue
            item_timestamp = parse_timestamp(data_item.get('timestamp'))
            data_item['_parsed_timestamp'] = item_timestamp # Store parsed timestamp

            # Check which of the *filtered* software are mentioned in this item
            for sw_name in all_software_names:
                 try:
                      pattern = re.compile(r'\b' + re.escape(sw_name) + r'\b', re.IGNORECASE)
                      if pattern.search(item_text):
                           # Store the entire data item if the software is mentioned
                           data_by_software[sw_name].append(data_item)
                 except re.error:
                      continue # Ignore if regex fails for a name

        # Calculate metrics for each software
        for software in software_mentions_detailed:
            software_name = software['name']
            mention_count = software['mentions'] # Use the count from extraction step
            platform_counts: Counter[str] = collections.Counter()
            engagement_metrics: Dict[str, float] = collections.defaultdict(float)
            mention_timestamps: List[datetime.datetime] = []

            # Iterate through data items where THIS software was mentioned
            relevant_data_items = data_by_software.get(software_name, [])

            for data in relevant_data_items:
                platform = data.get('platform', 'unknown')
                item_timestamp = data.get('_parsed_timestamp') # Use pre-parsed timestamp

                # Count platforms and collect timestamps
                platform_counts[platform] += 1 # Count item once per platform if mentioned
                if item_timestamp:
                    mention_timestamps.append(item_timestamp)

                # Calculate platform-specific engagement score for this item
                engagement_score = 0
                if platform == 'reddit':
                    score = data.get('upvotes', data.get('score', 0))
                    num_comments = len(data.get('comments', []))
                    engagement_score = (score if isinstance(score, (int, float)) else 0) + \
                                       (num_comments * 2 if isinstance(num_comments, int) else 0)
                elif platform == 'twitter':
                    likes = data.get('likes', data.get('favorite_count', 0))
                    retweets = data.get('retweets', data.get('retweet_count', 0))
                    replies = len(data.get('replies', []))
                    engagement_score = (likes if isinstance(likes, (int, float)) else 0) + \
                                       (retweets * 2 if isinstance(retweets, (int, float)) else 0) + \
                                       (replies * 3 if isinstance(replies, int) else 0)
                elif platform == 'youtube':
                    views = data.get('views', data.get('view_count', 0))
                    likes = data.get('likes', data.get('like_count', 0))
                    comments = len(data.get('comments', []))
                    safe_views = views if isinstance(views, (int, float)) else 0
                    safe_likes = likes if isinstance(likes, (int, float)) else 0
                    safe_comments = comments if isinstance(comments, int) else 0
                    # Log scale for views to prevent dominance, add likes and comments
                    engagement_score = math.log1p(safe_views / 1000 if safe_views > 0 else 0) + safe_likes + safe_comments * 2
                elif platform == 'producthunt':
                    upvotes = data.get('upvotes', 0)
                    comments = len(data.get('comments', []))
                    safe_upvotes = upvotes if isinstance(upvotes, (int, float)) else 0
                    safe_comments = comments if isinstance(comments, int) else 0
                    engagement_score = safe_upvotes * 1.5 + safe_comments * 2
                else: # Generic fallback: use comment count
                    comments = len(data.get('comments', []))
                    safe_comments = comments if isinstance(comments, int) else 0
                    engagement_score = safe_comments

                engagement_metrics[platform] += engagement_score

            # Calculate derived metrics after processing all relevant items
            total_mentions_for_sw = mention_count # Use the pre-calculated total mentions
            platform_mention_counts = collections.Counter()
            for data in relevant_data_items:
                 item_text = self._combine_text_data([data])
                 try:
                     pattern = re.compile(r'\b' + re.escape(software_name) + r'\b', re.IGNORECASE)
                     matches_in_item = len(pattern.findall(item_text))
                     platform_mention_counts[data.get('platform', 'unknown')] += matches_in_item
                 except re.error: continue

            # Use platform_mention_counts for distribution if more accurate counts needed per platform
            # For percentages, using total_mentions_for_sw derived from items count is simpler here:
            platform_item_counts = collections.Counter(d.get('platform', 'unknown') for d in relevant_data_items)
            total_items_mentioning_sw = len(relevant_data_items)
            platform_percentages = {
                platform: round(count / total_items_mentioning_sw * 100, 2) if total_items_mentioning_sw else 0
                for platform, count in platform_item_counts.items()
            }

            recency_metrics = {}
            if mention_timestamps:
                now = datetime.datetime.now(datetime.timezone.utc)
                mention_timestamps.sort(reverse=True)
                most_recent = mention_timestamps[0]
                days_since_most_recent = (now - most_recent).days
                # Normalize recency score (higher is better) - capped at 100
                recency_score = max(0.0, 100.0 - days_since_most_recent * 2.0) # Linear decay, 0 after 50 days

                # Calculate mentions per day over the observed period
                mentions_per_day = 0
                if len(mention_timestamps) >= 2:
                    oldest = mention_timestamps[-1]
                    # Ensure time span is at least 1 day to avoid division by zero
                    time_span_days = max(1.0, (most_recent - oldest).total_seconds() / (24 * 3600))
                    mentions_per_day = len(mention_timestamps) / time_span_days
                elif len(mention_timestamps) == 1:
                     mentions_per_day = 1.0 # Assume 1 mention if only one timestamp

                recency_metrics = {
                    'days_since_most_recent': days_since_most_recent,
                    'recency_score': round(recency_score, 2), # 0-100 scale
                    'mentions_per_day_observed': round(mentions_per_day, 3) # Avg over the period data was found
                }

            # Get average sentiment for demand score calculation
            avg_sentiment = sentiment_analysis.get(software_name, {}).get('average_sentiment', 0.0)

            # Calculate final demand score
            demand_score = self._calculate_demand_score(
                total_mentions_for_sw, # Use the accurate total mention count
                platform_item_counts, # Use counts of items per platform for diversity
                engagement_metrics,
                recency_metrics,
                avg_sentiment
            )

            metrics[software_name] = {
                'total_mentions': total_mentions_for_sw,
                'mention_percentage_overall': round(total_mentions_for_sw / total_all_mentions * 100, 2) if total_all_mentions > 1 else 100.0,
                'platform_distribution_items': dict(platform_item_counts), # Distribution of data items mentioning the sw
                'platform_distribution_mentions': dict(platform_mention_counts), # Distribution of actual mentions
                'platform_percentages_items': platform_percentages,
                'engagement_metrics_total': round(sum(engagement_metrics.values()), 2),
                'engagement_by_platform': {k: round(v, 2) for k,v in engagement_metrics.items()},
                'recency_metrics': recency_metrics,
                'demand_score': demand_score # Single combined score
            }

        # Sort final metrics dictionary by demand score descending
        return dict(sorted(metrics.items(), key=lambda item: item[1]['demand_score'], reverse=True))

    def _calculate_demand_score(self,
                              mention_count: int,
                              platform_item_counts: Counter[str],
                              engagement_metrics: Dict[str, float],
                              recency_metrics: Dict[str, Any],
                              avg_sentiment: float) -> float:
        """Calculate an enhanced demand score based on multiple factors."""
        if mention_count == 0:
            return 0.0

        # --- Weighting Factors (adjustable heuristics) ---
        W_MENTIONS = 10.0
        W_DIVERSITY = 0.5 # Max bonus for platform diversity
        W_ENGAGEMENT = 20.0 # Divisor for log engagement score (lower = higher impact)
        W_RECENCY = 0.5 # Max bonus for recency score
        W_FREQUENCY = 0.2 # Multiplier for log mentions/day bonus
        W_SENTIMENT = 0.5 # Max adjustment (+/-) based on sentiment (scaled)

        # 1. Base Score: Log of mentions to dampen effect of extremely high counts
        # Use total actual mentions for base score magnitude
        base_score = math.log1p(mention_count) * W_MENTIONS

        # 2. Platform Diversity: Bonus for appearing on more platforms (using item counts)
        num_platforms = len([count for count in platform_item_counts.values() if count > 0])
        # Normalize diversity score (0 to 1, assuming max ~5 platforms reasonable baseline)
        diversity_factor = 1.0 + min(1.0, num_platforms / 5.0) * W_DIVERSITY

        # 3. Engagement Factor: Log of total engagement (sum across platforms)
        total_engagement = sum(engagement_metrics.values())
        # Normalize engagement (log scale, higher engagement increases factor)
        engagement_factor = 1.0 + math.log1p(total_engagement) / W_ENGAGEMENT

        # 4. Recency Factor: Use the 0-100 recency score
        recency_score = recency_metrics.get('recency_score', 0)
        # Normalize recency (max W_RECENCY bonus for score of 100)
        recency_factor = 1.0 + (recency_score / 100.0) * W_RECENCY

        # 5. Frequency Factor: Average mentions per day over observed period (capped bonus)
        mentions_per_day = recency_metrics.get('mentions_per_day_observed', 0)
        # Normalize frequency (log scale, provides smaller bonus)
        frequency_factor = 1.0 + math.log1p(mentions_per_day) * W_FREQUENCY

        # 6. Sentiment Factor: Adjust score based on average sentiment.
        # Negative sentiment slightly increases score (implying problems needing solutions),
        # Positive sentiment slightly decreases it (implying satisfaction).
        # Map sentiment (-1 to 1) to a factor around 1.0.
        # Example: Scale sentiment impact between -0.25 and +0.25 -> factor 0.75 to 1.25
        sentiment_adjustment = -(avg_sentiment * W_SENTIMENT) # Negative sentiment gives positive adjustment
        sentiment_factor = 1.0 + sentiment_adjustment

        # Combine factors multiplicatively with the base score
        final_score = base_score * diversity_factor * engagement_factor * recency_factor * frequency_factor * sentiment_factor

        # Ensure score is non-negative
        return max(0.0, round(final_score, 2))


    def _generate_product_ideas(self, pain_points_data: List[Dict[str, Any]], feature_requests_data: List[Dict[str, Any]], demand_metrics: Dict[str, Any], co_occurrence: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
        """Generate product ideas based on pain points, features, demand, and co-occurrence using AI and fallbacks."""
        product_ideas = []
        processed_software_fallback = set() # Track software processed by fallback to avoid duplicates
        ai_generated_targets = set() # Track software targeted by AI ideas

        # Combine pain points and features into a single lookup map for easier access
        all_issues_map: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for item in pain_points_data + feature_requests_data:
            all_issues_map[item['software']].extend(item['issues']) # Assumes 'issues' key exists from _extract_pain_points...

        ai_available = bool(self.api_key and (self.genai or self.genai_client or self.openai or self.anthropic_client))

        # --- Attempt AI Idea Generation First (If API Key Available) ---
        if ai_available:
            try:
                logging.info(f"Attempting product idea generation using AI ({self.api_type})...")
                # Prepare concise input for the AI prompt: focus on top N software by demand
                top_demand_sw_names = list(demand_metrics.keys())[:10] # Focus on top 10 by demand score
                relevant_issues_summary = []
                for sw_name in top_demand_sw_names:
                    issues = all_issues_map.get(sw_name, [])
                    if not issues: continue

                    # Sort issues within the software
                    pps = sorted([i for i in issues if i['type'] == 'pain_point' and i.get('severity')],
                                 key=lambda x: PAIN_POINT_KEYWORDS.get(x['severity'], 3)) # Sort by severity index
                    frs = [i for i in issues if i['type'] == 'feature_request']

                    # Select top 3 of each type for the prompt
                    top_pps_contexts = [f"{p['severity']}: {p['context'][:120]}..." for p in pps[:3]]
                    top_frs_contexts = [f"{f['context'][:120]}..." for f in frs[:3]]

                    # Only include software if it has significant issues
                    if top_pps_contexts or top_frs_contexts:
                        relevant_issues_summary.append({
                            "software": sw_name,
                            "demand_score": demand_metrics.get(sw_name, {}).get('demand_score', 0),
                            "top_pain_points": top_pps_contexts,
                            "top_feature_requests": top_frs_contexts
                        })

                if not relevant_issues_summary:
                     logging.warning("No significant issues found for top demand software to feed to AI for idea generation.")
                     # Proceed to fallback directly

                else:
                    # Limit the size of the summary to avoid overly long prompts
                    max_prompt_issues = 15 # Limit total items sent to AI
                    if len(relevant_issues_summary) > max_prompt_issues:
                        logging.warning(f"Too many relevant issues ({len(relevant_issues_summary)}); truncating to top {max_prompt_issues} for AI prompt.")
                        relevant_issues_summary = relevant_issues_summary[:max_prompt_issues]

                    ai_prompt = f"""
                    You are a Product Manager analyzing user feedback from online discussions to brainstorm SPECIFIC and ACTIONABLE product ideas or feature improvements.
                    Based ONLY on the provided software issues (pain points and feature requests) and their demand scores (higher score = more discussion), generate several distinct product ideas.

                    Focus on:
                    1. New tools addressing unmet needs highlighted by feature requests for high-demand software.
                    2. Improved alternatives solving critical/major pain points of existing high-demand software.
                    3. Integrations or plugins bridging gaps suggested by co-occurrence or feature requests involving multiple tools (if applicable from context).

                    Provided Data (Summarized Issues for High-Demand Software):
                    ---
                    {json.dumps(relevant_issues_summary, indent=2)}
                    ---

                    For EACH product idea, provide:
                    - "target_software": [List of one or more existing software the idea relates to/competes with]
                    - "idea_name": A concise, marketable name for the new product/feature.
                    - "idea_description": A clear explanation (1-2 sentences) of what the product/feature does and WHICH specific pain points or feature requests it solves, referencing the input data.
                    - "key_features": [List of 3-5 bullet points outlining core functionalities needed to address the issues]
                    - "justification": Briefly explain the opportunity based on the input (e.g., "Addresses critical 'crash' pain point for high-demand 'Tool X'").

                    Generate multiple distinct ideas if possible. Prioritize ideas addressing critical/major pain points or frequent feature requests for software with high demand scores.

                    Format your response STRICTLY as a JSON array of objects, each object representing one product idea with the keys mentioned above ("target_software", "idea_name", "idea_description", "key_features", "justification").
                    Example JSON Array:
                    [
                      {{
                        "target_software": ["Software A"],
                        "idea_name": "Software A Stability Suite",
                        "idea_description": "An add-on for Software A designed to fix common 'crash' and 'freeze' issues reported by users, ensuring smoother workflows.",
                        "key_features": [
                          "Real-time performance monitoring.",
                          "Automated resource management.",
                          "Crash diagnostics reporting.",
                          "Compatibility checker for plugins.",
                          "Proactive stability alerts."
                        ],
                        "justification": "Addresses critical stability pain points ('crash', 'freeze') for high-demand 'Software A'."
                      }},
                      {{ ... another idea object ... }}
                    ]
                    ---
                    Respond only with the JSON array. Do not include any preamble or explanation outside the JSON structure.
                    """

                    ai_ideas_raw = None
                    model_to_use = 'models/gemini-1.5-flash-latest' # Default Gemini model

                    # --- AI Call Logic ---
                    if self.api_type == 'gemini' and (self.genai or self.genai_client):
                         if self.genai and hasattr(self.genai, 'GenerativeModel'):
                              logging.info(f"Using Gemini model '{model_to_use}' via configured genai module for idea generation.")
                              model = self.genai.GenerativeModel(model_to_use)
                              # Add safety settings appropriate for idea generation if needed
                              # safety_settings = [...]
                              response = model.generate_content(ai_prompt) #, safety_settings=safety_settings)
                              ai_ideas_raw = response.text
                         elif self.genai_client and hasattr(self.genai_client, 'generate_content'):
                              logging.warning(f"Using Gemini model '{model_to_use}' via genai.Client instance for idea generation.")
                              response = self.genai_client.generate_content(model=model_to_use, contents=ai_prompt) # Adapt if Client API differs
                              ai_ideas_raw = response.text
                         else:
                              logging.error("Gemini AI object not configured correctly for idea generation.")

                    elif self.api_type == 'openai' and self.openai:
                         logging.info("Using OpenAI model 'gpt-4o-mini' for idea generation.")
                         response = self.openai.chat.completions.create(
                             model='gpt-4o-mini',
                             response_format={'type': 'json_object'},
                             messages=[
                                 {'role': 'system', 'content': 'You generate product ideas based on software pain points and feature requests, returning a JSON array of idea objects.'},
                                 {'role': 'user', 'content': ai_prompt}
                             ]
                         )
                         ai_ideas_raw = response.choices[0].message.content

                    elif self.api_type == 'anthropic' and self.anthropic_client:
                         logging.info("Using Anthropic model 'claude-3-haiku-20240307' for idea generation.")
                         response = self.anthropic_client.messages.create(
                             model='claude-3-haiku-20240307',
                             max_tokens=3000, # Allow ample tokens for multiple ideas
                             messages=[{'role': 'user', 'content': ai_prompt}]
                         )
                         ai_ideas_raw = response.content[0].text
                    # --- End AI Call Logic ---

                    if ai_ideas_raw:
                        # Extract JSON array robustly from potential markdown code blocks
                        json_match = re.search(r'\[\s*\{.*?\}\s*\]', ai_ideas_raw, re.DOTALL)
                        if json_match:
                            try:
                                parsed_ideas = json.loads(json_match.group(0))
                                if isinstance(parsed_ideas, list):
                                    valid_ideas_count = 0
                                    for idea in parsed_ideas:
                                        # Validate structure of each idea object
                                        if isinstance(idea, dict) and all(k in idea for k in ['target_software', 'idea_name', 'idea_description', 'key_features', 'justification']):
                                            primary_target = idea['target_software'][0] if idea.get('target_software') else None
                                            demand_score = 0
                                            if primary_target and primary_target in demand_metrics:
                                                 demand_score = demand_metrics[primary_target].get('demand_score', 0)

                                            product_ideas.append({
                                                'target_software': idea['target_software'],
                                                'idea_name': idea['idea_name'],
                                                'idea_description': idea['idea_description'],
                                                'key_features': idea['key_features'],
                                                'justification': idea['justification'],
                                                'demand_score_primary_target': demand_score,
                                                'generation_method': 'ai'
                                            })
                                            # Track which software were targeted by AI ideas
                                            if isinstance(idea['target_software'], list):
                                                ai_generated_targets.update(idea['target_software'])
                                            valid_ideas_count += 1
                                        else:
                                            logging.warning(f"Skipping malformed AI idea object: {idea}")

                                    logging.info(f"Successfully generated and parsed {valid_ideas_count} product ideas using AI.")
                            except json.JSONDecodeError as e:
                                logging.warning(f"AI response for ideas looked like JSON but failed to parse: {e}. Raw match starts: {json_match.group(0)[:200]}...")
                        else:
                            logging.warning(f"Could not find JSON array '[]' in AI idea response. Raw output starts: {ai_ideas_raw[:200]}...")
                    else:
                         logging.warning(f"AI ({self.api_type}) returned no output for idea generation.")

            except Exception as e:
                logging.error(f"Error during AI product idea generation ({self.api_type}): {e}", exc_info=True)
                # Proceed to fallback even if AI fails

        # --- Fallback / Augmentation Idea Generation ---
        logging.info("Running fallback/augmentation for product idea generation...")

        # Iterate through software sorted by demand, focusing on those NOT covered by AI
        sorted_demand = sorted(demand_metrics.items(), key=lambda item: item[1]['demand_score'], reverse=True)
        ideas_generated_fallback = 0
        max_fallback_ideas = 25 # Limit the number of fallback ideas

        for sw_name, metrics in sorted_demand:
            if ideas_generated_fallback >= max_fallback_ideas:
                 logging.info(f"Reached fallback idea limit ({max_fallback_ideas}).")
                 break
            # Skip if AI already targeted this software OR if already processed by fallback
            if sw_name in ai_generated_targets or sw_name in processed_software_fallback:
                 continue
            if sw_name not in all_issues_map: # Skip if no issues were mapped
                 continue

            demand_score = metrics.get('demand_score', 0)
            issues = all_issues_map[sw_name]
            # Get top pain point (most severe)
            pain_points = sorted(
                [i for i in issues if i['type'] == 'pain_point' and i.get('severity')],
                key=lambda x: pain_severity_order.index(x['severity']) if x['severity'] in pain_severity_order else 99 # Sort by severity index
            )
            # Get representative feature requests
            feature_requests = [i for i in issues if i['type'] == 'feature_request']

            # Idea Type 1: Address Top Pain Point (if critical or major)
            if pain_points:
                top_pp = pain_points[0]
                # Generate idea only for significant pain points
                if top_pp['severity'] in ['critical', 'major']:
                    # Simplify context for name/desc
                    simple_context = re.sub(r'\s+\(Source:.*?\)', '', top_pp['context'])[:150]
                    keyword = top_pp['keywords'][0] if top_pp['keywords'] else 'issues'

                    idea_name = f"{sw_name} Reliability/Usability Fix"
                    idea_desc = f"Addresses the top '{top_pp['severity']}' pain point for {sw_name} related to '{keyword}' ({simple_context}...)."
                    key_features = [
                        f"Improved stability/performance targeting '{keyword}'.",
                        "Enhanced error handling/reporting.",
                        f"Streamlined workflow for tasks related to '{keyword}'.",
                        "User feedback mechanism focused on stability."
                    ]
                    product_ideas.append({
                        'target_software': [sw_name], 'idea_name': idea_name, 'idea_description': idea_desc,
                        'key_features': key_features,
                        'justification': f"Targets '{top_pp['severity']}' pain ({keyword}) for {sw_name} (Demand: {demand_score:.1f}). Found {len(pain_points)} pain points.",
                        'demand_score_primary_target': demand_score, 'generation_method': 'fallback_painpoint'
                    })
                    processed_software_fallback.add(sw_name) # Mark as processed
                    ideas_generated_fallback += 1
                    if ideas_generated_fallback >= max_fallback_ideas: break

            # Idea Type 2: Fulfill Top Feature Request (if not already processed)
            if feature_requests and sw_name not in processed_software_fallback:
                top_fr = feature_requests[0] # Just take the first one found for simplicity
                keyword = top_fr['keywords'][0] if top_fr['keywords'] else 'feature'
                # Simplify context
                simple_context = re.sub(r'\s+\(Source:.*?\)', '', top_fr['context'])[:150]

                idea_name = f"{sw_name} Extension: {keyword.capitalize()} Integration"
                idea_desc = f"Adds the requested '{keyword}' capability to {sw_name}, addressing demand like: {simple_context}..."
                key_features = [
                    f"Seamless implementation of '{keyword}' feature.",
                    f"Integration with existing {sw_name} workflows.",
                    "User configuration options for the new feature.",
                    "Documentation and examples for usage."
                ]
                product_ideas.append({
                    'target_software': [sw_name], 'idea_name': idea_name, 'idea_description': idea_desc,
                    'key_features': key_features,
                    'justification': f"Fulfills feature request '{keyword}' for {sw_name} (Demand: {demand_score:.1f}). Found {len(feature_requests)} requests.",
                    'demand_score_primary_target': demand_score, 'generation_method': 'fallback_feature'
                })
                processed_software_fallback.add(sw_name)
                ideas_generated_fallback += 1
                if ideas_generated_fallback >= max_fallback_ideas: break

            # Idea Type 3: Alternative based on High Co-occurrence (if not already processed)
            if sw_name in co_occurrence and sw_name not in processed_software_fallback:
                 # Find top co-occurring software for sw_name
                 competitors = sorted(co_occurrence[sw_name].items(), key=lambda item: item[1], reverse=True)
                 if competitors:
                      top_competitor, count = competitors[0]
                      # Only generate if competitor is also somewhat relevant and not already processed
                      if top_competitor in demand_metrics and top_competitor not in processed_software_fallback and top_competitor not in ai_generated_targets:
                           comp_demand = demand_metrics.get(top_competitor, {}).get('demand_score', 0)
                           # Only suggest if competitor has some demand and co-occurrence is significant
                           if comp_demand > 0 and count >= 2: # Require at least 2 co-occurrences
                                idea_name = f"Unified Workflow: {sw_name} + {top_competitor}"
                                idea_desc = f"A tool combining key features of {sw_name} and {top_competitor} to streamline workflows, potentially addressing issues mentioned when both are discussed (Co-occurrence: {count})."
                                key_features = [
                                    f"Core functionality inspired by {sw_name}.",
                                    f"Core functionality inspired by {top_competitor}.",
                                    "Seamless data flow / integration between feature sets.",
                                    "Focus on addressing shared pain points (if identifiable)."
                                ]
                                product_ideas.append({
                                    'target_software': [sw_name, top_competitor], 'idea_name': idea_name, 'idea_description': idea_desc,
                                    'key_features': key_features,
                                    'justification': f"High co-occurrence ({count}) between {sw_name} (D:{demand_score:.1f}) & {top_competitor} (D:{comp_demand:.1f}) suggests integration opportunity.",
                                    'demand_score_primary_target': demand_score, # Base on primary sw
                                    'generation_method': 'fallback_cooccurrence'
                                })
                                processed_software_fallback.add(sw_name)
                                processed_software_fallback.add(top_competitor) # Mark competitor too
                                ideas_generated_fallback += 1
                                if ideas_generated_fallback >= max_fallback_ideas: break

        logging.info(f"Generated {ideas_generated_fallback} product ideas using fallback methods.")

        # Sort all combined ideas (AI + fallback) by the demand score of the primary target software
        return sorted(product_ideas, key=lambda x: x.get('demand_score_primary_target', 0), reverse=True)


    def _analyze_sentiment(self, software_mentions_detailed: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform sentiment analysis on context snippets for each software using TextBlob."""
        sentiment_results = {}
        if not hasattr(self, 'sentiment_analyzer') or self.sentiment_analyzer is None or self.sentiment_analyzer is DummyTextBlob:
             logging.warning("Sentiment analyzer (TextBlob) not available or not installed. Skipping sentiment analysis.")
             return {}

        logging.info("Performing sentiment analysis on software mention contexts...")
        for software in software_mentions_detailed:
            software_name = software['name']
            contexts_with_meta = software['context']
            if not contexts_with_meta: continue

            sentiment_scores = []
            # Extract text part from context_with_meta for analysis
            for context_meta in contexts_with_meta:
                context_match = re.match(r"^(.*)\s+\(Source:.*,\s+Timestamp:.*\)$", context_meta, re.DOTALL)
                context_text = context_match.group(1).strip() if context_match else context_meta

                # Limit context length for analysis if necessary
                analysis_text = context_text[:1500] if len(context_text) > 1500 else context_text
                try:
                    blob = self.sentiment_analyzer(analysis_text)
                    sentiment_scores.append(blob.sentiment.polarity)
                except Exception as e:
                    logging.warning(f"Sentiment analysis failed for context snippet of {software_name}: {e}. Snippet: {analysis_text[:100]}...")
                    # Optionally append a neutral score or skip
                    # sentiment_scores.append(0.0)

            if sentiment_scores:
                try:
                    avg_sentiment = statistics.mean(sentiment_scores)
                    std_dev_sentiment = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0.0

                    # Define sentiment categories based on average polarity
                    if avg_sentiment >= 0.2: sentiment_category = 'Positive'
                    elif avg_sentiment > 0.05: sentiment_category = 'Mostly Positive'
                    elif avg_sentiment <= -0.2: sentiment_category = 'Negative'
                    elif avg_sentiment < -0.05: sentiment_category = 'Mostly Negative'
                    else: sentiment_category = 'Neutral'

                    # Calculate distribution across more granular bins
                    bins = {'Positive': 0, 'Mostly Positive': 0, 'Neutral': 0, 'Mostly Negative': 0, 'Negative': 0}
                    for score in sentiment_scores:
                         if score >= 0.2: bins['Positive'] += 1
                         elif score > 0.05: bins['Mostly Positive'] += 1
                         elif score < -0.2: bins['Negative'] += 1
                         elif score <= -0.05: bins['Mostly Negative'] += 1
                         else: bins['Neutral'] += 1

                    sentiment_results[software_name] = {
                        'average_sentiment': round(avg_sentiment, 3),
                        'std_dev_sentiment': round(std_dev_sentiment, 3),
                        'sentiment_category': sentiment_category,
                        'sentiment_distribution': bins,
                        'sentiment_scores_sample': [round(s, 2) for s in sentiment_scores[:20]], # Larger sample
                        'analysis_sample_size': len(sentiment_scores) # Number of contexts analyzed
                    }
                except statistics.StatisticsError:
                    logging.warning(f"Could not calculate sentiment statistics for {software_name} (likely no valid scores).")
                except Exception as e:
                    logging.error(f"Unexpected error calculating sentiment stats for {software_name}: {e}", exc_info=True)

        # Sort results by average sentiment ascending (most negative first)
        return dict(sorted(sentiment_results.items(), key=lambda item: item[1]['average_sentiment']))


    def _perform_statistical_analysis(self, scraped_data: List[Dict[str, Any]], software_mentions_detailed: List[Dict[str, Any]], issues_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on mentions, issues, and co-occurrence."""
        stats_results: Dict[str, Any] = {
            'overall': {}, 'mention_statistics': {}, 'issue_statistics': {}, 'co_occurrence': {}
        }
        num_data_points = len(scraped_data)
        if num_data_points == 0:
            logging.warning("Cannot perform statistical analysis: No data provided.")
            return stats_results
        if not software_mentions_detailed:
             logging.warning("Cannot perform statistical analysis: No software mentions identified.")
             # Provide basic overall stats anyway
             stats_results['overall'] = {'total_data_points': num_data_points}
             return stats_results

        logging.info("Performing statistical analysis...")

        # --- Overall Stats ---
        platform_distribution = collections.Counter(data.get('platform', 'unknown') for data in scraped_data)
        total_content_length = sum(len(self._combine_text_data([data])) for data in scraped_data)
        total_sw_mentions_count = sum(sw['mentions'] for sw in software_mentions_detailed)

        stats_results['overall'] = {
            'total_data_points': num_data_points,
            'total_unique_software_identified': len(software_mentions_detailed),
            'total_software_mentions': total_sw_mentions_count,
            'platform_distribution': dict(platform_distribution),
            'average_mentions_per_software': round(total_sw_mentions_count / len(software_mentions_detailed), 2) if software_mentions_detailed else 0,
            'total_content_length_chars': total_content_length,
            'average_content_length_chars_per_item': round(total_content_length / num_data_points, 2) if num_data_points else 0
        }

        # --- Mention Statistics ---
        mention_counts = [software['mentions'] for software in software_mentions_detailed]
        if mention_counts:
            try:
                mean_mentions = statistics.mean(mention_counts)
                median_mentions = statistics.median(mention_counts)
                # Use population stdev if considering all identified mentions as the population
                # pstdev = statistics.pstdev(mention_counts) if len(mention_counts) > 0 else 0.0
                # Use sample stdev if considering this a sample
                stdev_mentions = statistics.stdev(mention_counts) if len(mention_counts) > 1 else 0.0

                # Calculate quartiles using statistics.quantiles for robustness
                num_mentions = len(mention_counts)
                if num_mentions >= 4:
                    q1, _, q3 = statistics.quantiles(mention_counts, n=4)
                elif num_mentions > 0:
                    sorted_counts = sorted(mention_counts)
                    q1 = sorted_counts[0] # Approximation for small samples
                    q3 = sorted_counts[-1] # Approximation for small samples
                else:
                    q1 = q3 = 0

                iqr = q3 - q1
                # Define outlier thresholds (can be adjusted)
                upper_bound = q3 + 1.5 * iqr
                lower_bound = q1 - 1.5 * iqr
                outliers = [sw['name'] for sw, count in zip(software_mentions_detailed, mention_counts)
                            if count > upper_bound or count < lower_bound]


                stats_results['mention_statistics'] = {
                    'mean': round(mean_mentions, 2),
                    'median': median_mentions,
                    'standard_deviation': round(stdev_mentions, 2),
                    'min': min(mention_counts),
                    'max': max(mention_counts),
                    'q1': round(q1, 2), # First Quartile
                    'q3': round(q3, 2), # Third Quartile
                    'iqr': round(iqr, 2), # Interquartile Range
                    'outliers_by_mentions': outliers # List of software names considered outliers
                }
            except statistics.StatisticsError as e:
                logging.warning(f"Could not calculate mention statistics: {e}")
            except Exception as e:
                logging.error(f"Unexpected error calculating mention stats: {e}", exc_info=True)
        else:
             stats_results['mention_statistics'] = {'message': 'No mention counts available.'}


        # --- Issue Statistics ---
        total_issues = sum(item['total_issues'] for item in issues_data)
        total_pain_points = sum(item['pain_point_summary']['count'] for item in issues_data)
        total_feature_requests = sum(item['feature_request_summary']['count'] for item in issues_data)
        critical_pain_points = sum(item['pain_point_summary']['severity_counts'].get('critical', 0) for item in issues_data)
        major_pain_points = sum(item['pain_point_summary']['severity_counts'].get('major', 0) for item in issues_data)
        minor_pain_points = sum(item['pain_point_summary']['severity_counts'].get('minor', 0) for item in issues_data)


        stats_results['issue_statistics'] = {
            'total_issues_extracted': total_issues,
            'total_pain_points': total_pain_points,
            'total_feature_requests': total_feature_requests,
            'pain_point_severity_distribution': {
                'critical': critical_pain_points,
                'major': major_pain_points,
                'minor': minor_pain_points
            },
            'avg_issues_per_software_with_issues': round(total_issues / len(issues_data), 2) if issues_data else 0,
            'percentage_pain_points': round(total_pain_points / total_issues * 100, 1) if total_issues else 0,
            'percentage_feature_requests': round(total_feature_requests / total_issues * 100, 1) if total_issues else 0,
        }

        # --- Co-occurrence Analysis ---
        # Limit to top N software for performance, ensure N >= 2
        co_occurrence_limit = min(25, len(software_mentions_detailed)) # Analyze top 25 or fewer
        top_software_names = [sw['name'] for sw in software_mentions_detailed[:co_occurrence_limit]]
        co_occurrence_matrix: Dict[str, Dict[str, int]] = {name: collections.defaultdict(int) for name in top_software_names}

        if len(top_software_names) >= 2:
            logging.info(f"Calculating co-occurrence matrix for top {len(top_software_names)} software...")
            # Pre-compile regex patterns for efficiency
            patterns = {name: re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE) for name in top_software_names}

            for data_item in scraped_data:
                text = self._combine_text_data([data_item])
                if not text: continue
                text_lower = text.lower() # Lowercase once per item

                # Find which top software are present in this item
                present_software = [name for name, pattern in patterns.items() if pattern.search(text_lower)]

                # If 2 or more top software are present, increment their co-occurrence counts
                if len(present_software) >= 2:
                    for i in range(len(present_software)):
                        for j in range(i + 1, len(present_software)):
                            name1, name2 = present_software[i], present_software[j]
                            # Increment counts symmetrically
                            co_occurrence_matrix[name1][name2] += 1
                            co_occurrence_matrix[name2][name1] += 1
            logging.info("Co-occurrence calculation complete.")

        # Clean up empty entries and convert defaultdicts to dicts for final output
        final_co_occurrence = {
            name: dict(counts)
            for name, counts in co_occurrence_matrix.items() if counts # Only include software that co-occurred with others
        }
        stats_results['co_occurrence'] = final_co_occurrence

        return stats_results


    def _detect_trends(self, scraped_data: List[Dict[str, Any]], software_mentions_detailed: List[Dict[str, Any]], issues_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect trends over time for software mentions and potentially issue types (basic)."""
        trend_results: Dict[str, Any] = {'software_trends': {}, 'issue_trends': {}}
        if not scraped_data:
            logging.warning("Cannot detect trends: No data provided.")
            return trend_results

        logging.info("Detecting trends...")

        # Group data by date (YYYY-MM-DD) using valid timestamps
        date_grouped_data: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        valid_timestamps_found = False
        for data in scraped_data:
            timestamp = data.get('timestamp')
            if timestamp:
                try:
                    # Attempt robust parsing
                    dt_obj = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    date_str = dt_obj.date().strftime('%Y-%m-%d')
                    date_grouped_data[date_str].append(data)
                    valid_timestamps_found = True
                except (ValueError, TypeError):
                    pass # Skip data points with invalid timestamps

        if not valid_timestamps_found:
            logging.warning("No valid timestamps found in data. Cannot perform trend analysis.")
            return trend_results

        sorted_dates = sorted(date_grouped_data.keys())
        if len(sorted_dates) < 5: # Require at least 5 data points (days) for a meaningful trend line
             logging.info(f"Not enough distinct dates with data ({len(sorted_dates)}) for robust trend analysis (need at least 5). Skipping.")
             return trend_results

        # 1. Software Mention Trends (for top N software)
        logging.info("Calculating software mention trends...")
        trend_limit = 20 # Limit trend analysis to top N software
        top_software_for_trends = software_mentions_detailed[:trend_limit]

        # Pre-compile regex patterns for trend calculation
        trend_patterns = {sw['name']: re.compile(r'\b' + re.escape(sw['name']) + r'\b', re.IGNORECASE)
                          for sw in top_software_for_trends}

        for software in top_software_for_trends:
            software_name = software['name']
            pattern = trend_patterns[software_name]
            daily_mentions = []
            for date in sorted_dates:
                daily_count = 0
                for data in date_grouped_data[date]:
                    text = self._combine_text_data([data])
                    if not text: continue
                    try:
                        daily_count += len(pattern.findall(text))
                    except re.error:
                        continue # Should not happen with pre-compiled patterns
                daily_mentions.append({'date': date, 'mentions': daily_count})

            # Calculate linear trend if enough data points
            trend_results['software_trends'][software_name] = self._calculate_linear_trend(daily_mentions, 'mentions')

        # 2. Overall Issue Trends (Simplified: Total Pain Points & Feature Requests per day)
        # TODO: This is an approximation. Accurate trend needs issues mapped to specific dates.
        logging.info("Calculating overall issue trends (approximate by date)...")
        daily_issues_summary = []
        issue_keywords_flat: Dict[str, str] = {
            kw.lower(): cat for cat, kws in ALL_ISSUE_KEYWORDS.items() for kw in kws
        }

        for date in sorted_dates:
            daily_pain_count = 0
            daily_feature_count = 0
            for data in date_grouped_data[date]:
                text = self._combine_text_data([data])
                if not text: continue
                text_lower = text.lower()
                # Check for any issue keyword presence in the day's text
                found_pain = False
                found_feature = False
                for keyword, category in issue_keywords_flat.items():
                     try:
                         if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                              if category == 'request': found_feature = True
                              else: found_pain = True
                              # Optimize: break if both types found for this item
                              if found_pain and found_feature: break
                     except re.error: continue

                if found_pain: daily_pain_count += 1 # Count item if any pain keyword found
                if found_feature: daily_feature_count += 1 # Count item if any feature keyword found

            daily_issues_summary.append({
                'date': date,
                'pain_point_items': daily_pain_count, # Items with potential pain points
                'feature_request_items': daily_feature_count # Items with potential features
            })

        # Calculate trends for the aggregated daily counts
        trend_results['issue_trends']['pain_point_activity'] = self._calculate_linear_trend(daily_issues_summary, 'pain_point_items')
        trend_results['issue_trends']['feature_request_activity'] = self._calculate_linear_trend(daily_issues_summary, 'feature_request_items')


        return trend_results

    def _calculate_linear_trend(self, time_series_data: List[Dict[str, Any]], value_key: str) -> Dict[str, Any]:
        """Calculates linear trend slope and direction using simple linear regression."""
        # Need at least 3 points to attempt a trend line
        n = len(time_series_data)
        default_result = {'trend_direction': 'insufficient_data', 'slope': 0.0, 'r_squared': 0.0, 'data_points': n}
        if n < 3:
            return default_result

        y = [point.get(value_key, 0) for point in time_series_data] # Default to 0 if key missing
        x = list(range(n)) # Time index (0, 1, 2, ...)

        try:
            # Calculate sums needed for linear regression
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_xx = sum(xi * xi for xi in x)
            sum_yy = sum(yi * yi for yi in y)

            # Calculate slope (m) and intercept (b) of y = mx + b
            denominator = (n * sum_xx - sum_x * sum_x)
            if denominator == 0:
                 # Avoid division by zero; happens if all x values are the same (unlikely here)
                 # Or if n=1 (already handled)
                 slope = 0.0
                 intercept = sum_y / n if n > 0 else 0.0
            else:
                 slope = (n * sum_xy - sum_x * sum_y) / denominator
                 intercept = (sum_y - slope * sum_x) / n

            # Calculate R-squared (coefficient of determination) to assess fit quality
            y_mean = sum_y / n
            ss_total = sum((yi - y_mean) ** 2 for yi in y)
            ss_residual = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

            if ss_total == 0: # Handle case where all y values are the same
                r_squared = 1.0 if ss_residual < 1e-10 else 0.0 # Perfect fit if residuals are zero
            else:
                r_squared = 1.0 - (ss_residual / ss_total)
                r_squared = max(0.0, min(1.0, r_squared)) # Clamp between 0 and 1

            # Determine trend direction based on slope and R-squared
            trend_direction = 'stable'
            # Require a minimum R-squared for trend to be considered significant
            r_squared_threshold = 0.1 # Adjust as needed (e.g., 0.1 or higher)
            # Require slope to be significantly different from zero relative to data scale? (optional)
            # slope_threshold = abs(y_mean * 0.01) if y_mean != 0 else 0.01 # e.g., 1% of mean value

            if r_squared >= r_squared_threshold: # Check if the fit is decent
                if slope > 0: # Consider slope magnitude later if needed
                    trend_direction = 'increasing'
                elif slope < 0:
                    trend_direction = 'decreasing'
                # else slope is near zero, keep 'stable'

            return {
                'trend_direction': trend_direction,
                'slope': round(slope, 4), # Increase precision for slope
                'intercept': round(intercept, 4),
                'r_squared': round(r_squared, 3), # Goodness of fit (0 to 1)
                'data_points': n,
                # Optionally include raw data for plotting:
                # 'daily_values': time_series_data
            }

        except ZeroDivisionError:
             logging.warning(f"Division by zero encountered during trend calculation for key '{value_key}'.")
             return {**default_result, 'trend_direction': 'error_division_by_zero'}
        except Exception as e:
            logging.error(f"Error calculating linear trend for key '{value_key}': {e}", exc_info=True)
            return {**default_result, 'trend_direction': f'error_{type(e).__name__}'}


    def _generate_ai_insights(self, software_mentions: List[Dict[str, Any]], pain_points: List[Dict[str, Any]], feature_requests: List[Dict[str, Any]], sentiment_analysis: Dict[str, Any], statistical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive AI insights using the connected AI API based on analysis results."""
        # Check if AI is available and initialized
        ai_available = bool(self.api_key and (self.genai or self.genai_client or self.openai or self.anthropic_client))
        if not ai_available:
            return {'error': 'AI insights require a configured and successfully initialized AI API connection'}

        logging.info(f"Preparing data for AI insight generation ({self.api_type})...")

        # --- Prepare Concise Summaries for the Prompt ---
        # Use detailed mentions list passed as argument
        top_sw_mentions_summary = [{'name': sw['name'], 'mentions': sw['mentions']} for sw in software_mentions[:10]]

        # Use demand metrics stored in self.demand_metrics
        top_demand_sw_summary = sorted(
            [{'name': k, 'demand_score': v.get('demand_score', 0)} for k, v in self.demand_metrics.items()],
            key=lambda x: x['demand_score'], reverse=True
        )[:10]

        # Summarize top pain points (more detail)
        top_pain_points_summary = []
        for pp_group in pain_points[:7]: # Focus on top 7 software with pain points
            summary = {
                "software": pp_group['software'],
                "total_pain_points": pp_group['pain_point_summary']['count'],
                "severity_counts": pp_group['pain_point_summary']['severity_counts'],
                 # Include a few context examples, simplified
                "top_issue_examples": [
                    f"{iss['severity']}: {re.sub(r'\\s+\\(Source:.*?\\)', '', iss['context'])[:100]}..."
                    for iss in pp_group['issues']
                    if iss['type']=='pain_point' and iss.get('severity')
                ][:3] # Top 3 examples
            }
            top_pain_points_summary.append(summary)

        # Summarize top feature requests (more detail)
        top_feature_requests_summary = []
        # Sort software by feature request count
        fr_counts = {fr['software']: fr['feature_request_summary']['count'] for fr in feature_requests if fr['feature_request_summary']['count'] > 0}
        top_fr_sw_names = sorted(fr_counts, key=fr_counts.get, reverse=True)[:7] # Top 7 software by FR count

        for sw_name in top_fr_sw_names:
             # Find the corresponding feature request group
             fr_group = next((fr for fr in feature_requests if fr['software'] == sw_name), None)
             if fr_group:
                 requests_examples = [
                     f"{re.sub(r'\\s+\\(Source:.*?\\)', '', iss['context'])[:100]}..."
                     for iss in fr_group['issues'] if iss['type']=='feature_request'
                 ][:3] # Top 3 examples

                 summary = {
                      "software": sw_name,
                      "request_count": fr_group['feature_request_summary']['count'],
                      "top_request_examples": requests_examples
                 }
                 top_feature_requests_summary.append(summary)


        # Sentiment Highlights (already sorted by _analyze_sentiment)
        positive_sentiment_summary = {
            k: {'avg_sentiment': v['average_sentiment'], 'category': v['sentiment_category']}
            for k, v in list(reversed(list(sentiment_analysis.items())))[:5] # Top 5 most positive
            if v['average_sentiment'] > 0.05
        }
        negative_sentiment_summary = {
            k: {'avg_sentiment': v['average_sentiment'], 'category': v['sentiment_category']}
            for k, v in list(sentiment_analysis.items())[:5] # Top 5 most negative (already sorted this way)
            if v['average_sentiment'] < -0.05
        }

        # Statistical Summary (simplify for prompt)
        stats_summary = {
            "overall": {
                "total_data_points": statistical_analysis.get("overall", {}).get("total_data_points"),
                "total_unique_software": statistical_analysis.get("overall", {}).get("total_unique_software_identified"),
                "total_mentions": statistical_analysis.get("overall", {}).get("total_software_mentions"),
            },
            "issue_stats": {
                "total_issues": statistical_analysis.get("issue_statistics", {}).get("total_issues_extracted"),
                "perc_pain_points": statistical_analysis.get("issue_statistics", {}).get("percentage_pain_points"),
                "critical_pain_points": statistical_analysis.get("issue_statistics", {}).get("pain_point_severity_distribution", {}).get("critical"),
            },
            # Select top 3 co-occurring pairs based on count
            "top_cooccurring_pairs": sorted(
                 [(k, partner, count) for k, partners in statistical_analysis.get("co_occurrence", {}).items() for partner, count in partners.items() if k < partner], # Avoid duplicates
                 key=lambda x: x[2], reverse=True
             )[:3] # Top 3 pairs by count
        }

        # --- Construct the AI Prompt ---
        prompt = f"""
        Analyze the following consolidated data from online discussions about software tools. Provide a concise market analysis and strategic insights report, formatted as a JSON object.

        **Input Data Summary:**

        1.  **Top Mentioned Software:** {json.dumps(top_sw_mentions_summary, indent=2)}
        2.  **Top Software by Demand Score:** {json.dumps(top_demand_sw_summary, indent=2)}
        3.  **Software with Most/Critical Pain Points:** {json.dumps(top_pain_points_summary, indent=2)}
        4.  **Software with Most Feature Requests:** {json.dumps(top_feature_requests_summary, indent=2)}
        5.  **Sentiment Highlights:**
            - Most Positively Viewed: {json.dumps(positive_sentiment_summary, indent=2)}
            - Most Negatively Viewed: {json.dumps(negative_sentiment_summary, indent=2)}
        6.  **Statistical Highlights:** {json.dumps(stats_summary, indent=2)}

        **Analysis Report Sections (Respond ONLY with a valid JSON object containing these keys):**

        1.  `market_overview` (string): Briefly describe the software landscape based on mentions, demand. What types of tools are popular?
        2.  `key_pain_point_themes` (list of strings): Identify 2-4 major recurring categories of problems users face (e.g., "Performance/Stability", "Usability/UX", "Cost/Pricing", "Integration Issues"). Justify briefly with examples from the data summary.
        3.  `top_feature_demand_themes` (list of strings): What are 2-4 major categories of features users want (e.g., "AI Features", "Collaboration Enhancements", "API/Integrations", "Customization")? Justify briefly.
        4.  `opportunity_assessment` (list of objects): Identify the 2-4 most promising opportunities for new products or improvements based *only* on the provided data (high demand + issues/requests). Each object should have: `{"opportunity": "Brief description", "target_software": ["Relevant SW"], "justification": "Link to specific pain/request & demand"}`.
        5.  `competitive_insights` (string): Analyze relationships. Clear competitors? Users comparing tools (based on co-occurrence)? Tools with very negative sentiment likely losing users?
        6.  `sentiment_summary` (string): Overall user sentiment trend? Tools with polarized views?
        7.  `strategic_recommendations` (list of strings): Provide 2-3 actionable recommendations for developers based *directly* on the analysis (e.g., "Focus on improving stability for Tool X", "Develop integrations for Tool Y").
        8.  `potential_risks_or_gaps` (string): Briefly note risks (e.g., dominant players) or data gaps suggested by the analysis.

        **Strictly format your entire response as a single, valid JSON object.**
        ```json
        {{
          "market_overview": "...",
          "key_pain_point_themes": ["Theme 1: Description...", "Theme 2: Description..."],
          "top_feature_demand_themes": ["Theme A: Description...", "Theme B: Description..."],
          "opportunity_assessment": [
            {{"opportunity": "...", "target_software": ["..."], "justification": "..."}},
            {{...}}
          ],
          "competitive_insights": "...",
          "sentiment_summary": "...",
          "strategic_recommendations": ["Rec 1: ...", "Rec 2: ..."],
          "potential_risks_or_gaps": "..."
        }}
        ```
        """

        logging.info(f"Sending prompt (approx length: {len(prompt)} chars) to {self.api_type} for insights...")
        ai_insights_result: Dict[str, Any] = {'error': f'AI analysis failed to generate using {self.api_type}'} # Default error

        try:
            # --- Execute AI Call ---
            raw_ai_output = None
            model_to_use = 'models/gemini-1.5-flash-latest' # Default Gemini model

            if self.api_type == 'gemini' and (self.genai or self.genai_client):
                if self.genai and hasattr(self.genai, 'GenerativeModel'):
                    logging.info(f"Using Gemini model '{model_to_use}' via configured genai module for insights.")
                    model = self.genai.GenerativeModel(model_to_use)
                    # Consider adding specific safety settings if generating sensitive content summaries
                    response = model.generate_content(prompt) #, safety_settings=...)
                    raw_ai_output = response.text
                elif self.genai_client and hasattr(self.genai_client, 'generate_content'):
                    logging.warning(f"Using Gemini model '{model_to_use}' via genai.Client instance for insights.")
                    response = self.genai_client.generate_content(model=model_to_use, contents=prompt) # Adapt call if needed
                    raw_ai_output = response.text
                else:
                    logging.error("Gemini AI object not configured correctly for insights generation.")
                    ai_insights_result['error'] = "Gemini AI object not configured correctly."

            elif self.api_type == 'openai' and self.openai:
                logging.info("Using OpenAI model 'gpt-4o-mini' for insights.")
                response = self.openai.chat.completions.create(
                     model="gpt-4o-mini", # Good balance for structured JSON output
                     response_format={"type": "json_object"},
                     messages=[
                         {'role': 'system', 'content': 'You analyze software market data and generate strategic insights as a single JSON object.'},
                         {'role': 'user', 'content': prompt}
                     ],
                     temperature=0.5 # Slightly more deterministic for JSON structure
                 )
                raw_ai_output = response.choices[0].message.content

            elif self.api_type == 'anthropic' and self.anthropic_client:
                logging.info("Using Anthropic model 'claude-3-haiku-20240307' for insights.")
                response = self.anthropic_client.messages.create(
                     model="claude-3-haiku-20240307", # Fast model
                     max_tokens=3500, # Allow ample space for detailed JSON response
                     messages=[{'role': 'user', 'content': prompt}],
                     temperature=0.5
                 )
                raw_ai_output = response.content[0].text

            # --- Process AI Response ---
            if raw_ai_output:
                logging.info(f"Received raw response from {self.api_type} (length: {len(raw_ai_output)} chars). Attempting to parse JSON...")
                # Extract JSON object robustly (handles potential markdown ```json ... ``` fences)
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_ai_output, re.DOTALL | re.IGNORECASE)
                if not json_match: # Fallback to finding any JSON object if no markdown fence
                     json_match = re.search(r'(\{.*?\})', raw_ai_output, re.DOTALL | re.IGNORECASE)

                if json_match:
                    extracted_json_str = json_match.group(1)
                    try:
                        parsed_json = json.loads(extracted_json_str)
                        # Basic validation: Check if it's a dictionary and contains expected top-level keys
                        required_keys = ["market_overview", "key_pain_point_themes", "top_feature_demand_themes",
                                         "opportunity_assessment", "competitive_insights", "sentiment_summary",
                                         "strategic_recommendations", "potential_risks_or_gaps"]
                        if isinstance(parsed_json, dict) and all(key in parsed_json for key in required_keys):
                             logging.info(f"Successfully parsed valid JSON insights from {self.api_type}.")
                             return parsed_json # Success! Return the parsed dictionary
                        else:
                             missing_keys = [key for key in required_keys if key not in parsed_json]
                             logging.warning(f"AI response JSON is missing required keys: {missing_keys}")
                             ai_insights_result = {'error': f"AI response missing keys: {missing_keys}", 'raw_response_snippet': extracted_json_str[:500]}

                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to decode JSON from AI response: {e}. Raw matched JSON starts: {extracted_json_str[:500]}...")
                        ai_insights_result = {'error': f'JSONDecodeError: {e}', 'raw_response_snippet': extracted_json_str[:500]}
                else:
                    logging.warning("Could not find any JSON object structure (`{...}`) in the AI response.")
                    ai_insights_result = {'error': 'Could not find JSON object in AI response.', 'raw_response_snippet': raw_ai_output[:500]}
            else:
                 logging.warning(f"Received no output from {self.api_type} for AI insights.")
                 ai_insights_result['error'] = f"No output received from {self.api_type}."

        except Exception as e:
            logging.error(f"An error occurred generating AI insights with {self.api_type}: {e}", exc_info=True)
            ai_insights_result = {'error': f"Failed to generate AI insights with {self.api_type}: {str(e)}"}

        return ai_insights_result # Return dictionary possibly containing an error


    def generate_ai_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive AI analysis based on previously computed results.
        This acts as a public wrapper around _generate_ai_insights using the results dict.

        Args:
            analysis_results: The dictionary returned by the `analyze_data` method.

        Returns:
            A dictionary containing the AI-generated insights, or an error message.
        """
        ai_available = bool(self.api_key and (self.genai or self.genai_client or self.openai or self.anthropic_client))
        if not ai_available:
            return {'error': 'AI analysis requires a configured and successfully initialized AI API connection'}

        # Extract necessary data from the results dictionary
        # Use detailed mentions if available, otherwise fallback to simplified list
        software_mentions_det = analysis_results.get('software_mentions_detailed', analysis_results.get('software_mentions', [])) # Attempt to find detailed list if stored
        pain_points = analysis_results.get('pain_points', [])
        feature_requests = analysis_results.get('feature_requests', [])
        sentiment_analysis = analysis_results.get('sentiment_analysis', {})
        statistical_analysis = analysis_results.get('statistical_analysis', {})

        # Ensure demand metrics are available on the instance if needed by the internal method
        # (Current implementation accesses self.demand_metrics directly)
        self.demand_metrics = analysis_results.get('demand_metrics', self.demand_metrics) # Update if provided

        logging.info("Calling internal _generate_ai_insights with provided analysis results...")
        ai_analysis_results = self._generate_ai_insights(
            software_mentions_det, # Pass the most detailed list available
            pain_points,
            feature_requests,
            sentiment_analysis,
            statistical_analysis
        )
        return ai_analysis_results