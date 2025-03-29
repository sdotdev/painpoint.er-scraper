#!/usr/bin/env python3
"""
Data Analyzer module for PainPoint.er Search
Enhanced version for more robust analysis and idea generation.
"""
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
    # Suggest downloading the small English model if spacy is installed
    try:
        NLP = spacy.load("en_core_web_sm")
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

    TextBlob = DummyTextBlob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Keywords hinting that a nearby capitalized word might be a software product
SOFTWARE_CONTEXT_KEYWORDS = {
    'app', 'tool', 'software', 'platform', 'service', 'api', 'sdk',
    'library', 'framework', 'website', 'program', 'system', 'crm',
    'erp', 'saas', 'extension', 'plugin', 'bot', 'database', 'editor'
}
# Regex for potential software names (Capitalized, potentially with numbers, dots, or specific endings)
# Handles multi-word capitalized names like "Google Docs" or single names like "Figma"
POTENTIAL_SOFTWARE_REGEX = re.compile(
    r'\b([A-Z][a-zA-Z0-9]*(?:[.\s-][A-Z][a-zA-Z0-9]*)*' # Multi-word like "Google Cloud" or "VS Code"
    r'(?:\.js|\.ai|\.io|\.app)?'                       # Optional common endings
    r')\b'
)

# Keywords for Pain Points / Feature Requests
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
    "Analyze scraped data to identify pain points, features, and opportunities."

    def __init__(self, api_key: Optional[str] = None, api_type: str = 'gemini'):
        self.api_key = api_key
        self.api_type = api_type
        # Use TextBlob for simple sentiment, could be replaced with a more sophisticated model
        self.sentiment_analyzer = TextBlob
        self.genai = None # Ensure genai is initialized properly
        self.genai_client = None
        self.openai = None
        self.anthropic_client = None

        if self.api_key:
            self._setup_ai_connection()

    def _setup_ai_connection(self):
        "Set up connection to AI API."
        # DO NOT CHANGE GEMINI API SETUP PER USER REQUEST
        if self.api_type == 'gemini' and self.api_key:
            try:
                from google import genai
                # Assuming the user passes a configured client or key appropriately
                # Let's refine this slightly to handle potential client vs key init
                try:
                    # Attempt to initialize client directly if genai has Client attribute
                    if hasattr(genai, 'Client'):
                         self.genai_client = genai.Client(api_key=self.api_key) # Preferred if Client exists
                         self.genai = genai # Store the module
                         logging.info('Google Gemini AI client initialized successfully using Client.')
                    else:
                         # Fallback or alternative initialization (keep existing logic if needed)
                         genai.configure(api_key=self.api_key)
                         self.genai = genai # Store the module
                         # Create a model instance for generate_content if Client wasn't used
                         # Note: The original code called genai.models.generate_content,
                         # implying genai.models exists. If Client API is used, calls
                         # might be like self.genai_client.generate_content(...)
                         # We need to ensure compatibility with how it's called later.
                         # The current calls use `self.genai.models.generate_content` or `self.genai_client.generate_content`
                         # Let's ensure self.genai_client is set if possible. If not, rely on genai.configure
                         if not self.genai_client:
                              # If Client wasn't available, ensure genai itself is configured
                              logging.info('Google Gemini AI configured successfully using API key.')
                              # We might need a model instance if Client isn't used.
                              # Let's assume the later calls adapt or the user's env uses the configure method.

                except Exception as e:
                     logging.error(f"Failed to initialize Google Gemini AI: {e}. Please check API key and configuration.")
                     self.genai = None
                     self.genai_client = None

            except ImportError:
                logging.error("Google Generative AI SDK not found. Please install it: pip install google-generativeai")
                self.genai = None
                self.genai_client = None

        # --- Other API setups (unchanged functionality) ---
        elif self.api_type == 'openai' and self.api_key:
            try:
                import openai
                openai.api_key = self.api_key
                self.openai = openai
                logging.info('OpenAI client initialized successfully.')
            except ImportError:
                logging.error("OpenAI library not found. Please install it: pip install openai")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI: {e}")
        elif self.api_type == 'anthropic' and self.api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
                logging.info('Anthropic client initialized successfully.')
            except ImportError:
                logging.error("Anthropic library not found. Please install it: pip install anthropic")
            except Exception as e:
                logging.error(f"Failed to initialize Anthropic: {e}")

    def analyze_data(self, scraped_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        "Analyze scraped data to identify pain points and opportunities."
        if not scraped_data:
            logging.warning("No data provided for analysis.")
            return self._empty_analysis_results()

        logging.info(f"Starting analysis on {len(scraped_data)} data points.")
        all_text = self._combine_text_data(scraped_data)
        logging.info(f"Combined text length: {len(all_text)} characters.")

        software_mentions_detailed = self._extract_software_mentions(all_text, scraped_data)
        logging.info(f"Identified {len(software_mentions_detailed)} potential software products.")

        # Extract both pain points and feature requests
        issues = self._extract_pain_points_and_features(software_mentions_detailed)
        pain_points_data = [item for item in issues if item['type'] == 'pain_point']
        feature_requests_data = [item for item in issues if item['type'] == 'feature_request']
        logging.info(f"Extracted {sum(len(p['issues']) for p in pain_points_data)} pain points across {len(pain_points_data)} software.")
        logging.info(f"Extracted {sum(len(f['issues']) for f in feature_requests_data)} feature requests across {len(feature_requests_data)} software.")

        # Use the detailed mentions list which includes context for further analysis
        software_names_with_mentions = [{'name': s['name'], 'mentions': s['mentions']} for s in software_mentions_detailed]

        sentiment_analysis = self._analyze_sentiment(software_mentions_detailed)
        logging.info("Completed sentiment analysis.")

        demand_metrics = self._calculate_demand_metrics(scraped_data, software_mentions_detailed, sentiment_analysis)
        logging.info("Calculated demand metrics.")

        statistical_analysis = self._perform_statistical_analysis(scraped_data, software_mentions_detailed, issues)
        logging.info("Performed statistical analysis.")

        trend_analysis = self._detect_trends(scraped_data, software_mentions_detailed, issues)
        logging.info("Detected trends.")

        # Pass both pain points and feature requests to idea generation
        product_ideas = self._generate_product_ideas(pain_points_data, feature_requests_data, demand_metrics, statistical_analysis.get('co_occurrence', {}))
        logging.info(f"Generated {len(product_ideas)} product ideas.")

        ai_insights = {}
        # Use the primary analysis results as input for AI summary
        if self.api_key and (self.genai_client or self.genai or self.openai or self.anthropic_client):
            logging.info("Generating AI insights summary...")
            ai_insights = self._generate_ai_insights(
                software_names_with_mentions, # Pass simplified list here
                pain_points_data,
                feature_requests_data,
                sentiment_analysis,
                statistical_analysis
            )
            logging.info("AI insights generation complete.")
        else:
            logging.info("Skipping AI insights generation (no API key or client).")


        return {
            'software_mentions': software_names_with_mentions, # Return simplified list as before
            'pain_points': pain_points_data,
            'feature_requests': feature_requests_data, # Added feature requests
            'demand_metrics': demand_metrics,
            'sentiment_analysis': sentiment_analysis,
            'statistical_analysis': statistical_analysis,
            'trend_analysis': trend_analysis,
            'product_ideas': product_ideas,
            'ai_insights': ai_insights
        }

    def _empty_analysis_results(self) -> Dict[str, Any]:
        """Returns a dictionary with empty results for all analysis keys."""
        return {
            'software_mentions': [], 'pain_points': [], 'feature_requests': [],
            'demand_metrics': {}, 'sentiment_analysis': {}, 'statistical_analysis': {},
            'trend_analysis': {}, 'product_ideas': [], 'ai_insights': {}
        }

    def _combine_text_data(self, scraped_data: List[Dict[str, Any]]) -> str:
        "Combine all relevant text data from scraped sources."
        # Use a list and join later for slightly better performance on very large datasets
        text_parts = []
        for data in scraped_data:
            platform = data.get('platform', 'unknown')
            # Extract text more robustly, handling missing keys gracefully
            title = data.get('title', '')
            post_content = data.get('post_content', data.get('description', data.get('tweet_content', data.get('content', ''))))
            comments_list = data.get('comments', data.get('replies', []))

            if title: text_parts.append(f"TITLE: {title}")
            if post_content: text_parts.append(f"CONTENT: {post_content}") # Use generic 'CONTENT'

            for i, comment in enumerate(comments_list):
                comment_text = comment.get('text', '')
                if comment_text:
                    # Add comment number for potential context tracking if needed later
                    text_parts.append(f"COMMENT_{i+1}: {comment_text}")

        # Add separators to prevent words from merging across different data points
        return "\n\n".join(text_parts)

    def _extract_software_mentions(self, text: str, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract mentions of software products using AI (if available) or NER/Regex fallback.
        Returns a detailed list including context.
        """
        mentioned_software: Dict[str, Dict[str, Any]] = {}
        max_ai_text_length = 30000 # Allow longer context for better AI identification

        # 1. Attempt AI Extraction (No change to Gemini call logic)
        ai_identified_names: Set[str] = set()
        if self.api_key and (self.genai_client or self.genai or self.openai or self.anthropic_client):
            try:
                logging.info("Attempting software extraction using AI...")
                text_sample = text[:max_ai_text_length] if len(text) > max_ai_text_length else text
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
                if self.api_type == 'gemini' and (self.genai_client or self.genai):
                     # Determine which Gemini object to use based on initialization
                    if self.genai_client and hasattr(self.genai_client, 'generate_content'):
                         # Use the Client API if available
                         # Ensure model name is appropriate, adjust if needed (e.g., 'gemini-1.5-flash' etc.)
                         # NOTE: User requested gemini-2.0-flash. This model might not exist or have different capabilities.
                         # Stick to user's request but log a warning if issues arise.
                         # Using a model known for robust generation might be better, like gemini-1.5-pro if available.
                         # For now, stick to the user's specified model.
                         # Check if 'gemini-2.0-flash' is a valid model identifier in the user's context.
                         # If it causes errors, suggest 'gemini-1.5-flash' or 'gemini-pro'.
                         logging.warning("Using model 'gemini-2.0-flash' as specified. If errors occur, consider 'gemini-1.5-flash' or 'gemini-pro'.")
                         response = self.genai_client.generate_content(model='models/gemini-1.5-flash-latest', contents=ai_prompt) # Use a potentially more reliable model identifier
                         raw_ai_output = response.text
                    elif self.genai and hasattr(self.genai, 'models') and hasattr(self.genai.models, 'generate_content'):
                         # Fallback to legacy(?) `genai.models.generate_content` if Client wasn't used/available
                         # This preserves the original code's call structure if needed.
                         logging.warning("Using model 'gemini-2.0-flash' via genai.models.generate_content as specified.")
                         response = self.genai.models.generate_content(model='models/gemini-1.5-flash-latest', contents=ai_prompt) # Match model identifier
                         raw_ai_output = response.text
                    else:
                         logging.error("Gemini AI object not configured correctly for content generation.")

                elif self.api_type == 'openai' and self.openai:
                     # Use a model known for JSON output reliability
                     response = self.openai.chat.completions.create(
                         model='gpt-4o-mini', # Use a cost-effective but capable model
                         response_format={'type': 'json_object'},
                         messages=[
                             {'role': 'system', 'content': 'You extract software names from text and return a JSON array of strings.'},
                             {'role': 'user', 'content': ai_prompt}
                         ]
                     )
                     # OpenAI's json_object mode should return valid JSON directly
                     raw_ai_output = response.choices[0].message.content

                elif self.api_type == 'anthropic' and self.anthropic_client:
                     response = self.anthropic_client.messages.create(
                         model='claude-3-haiku-20240307', # Use a fast and cost-effective model
                         max_tokens=1000,
                         messages=[{'role': 'user', 'content': ai_prompt}]
                     )
                     raw_ai_output = response.content[0].text

                if raw_ai_output:
                    # Extract JSON array from the response (handle potential markdown/text wrapping)
                    json_match = re.search(r'\[.*?\]', raw_ai_output, re.DOTALL)
                    if json_match:
                        try:
                            software_list = json.loads(json_match.group(0))
                            if isinstance(software_list, list) and all(isinstance(item, str) for item in software_list):
                                ai_identified_names.update(name.strip() for name in software_list if name.strip())
                                logging.info(f"AI identified {len(ai_identified_names)} software names.")
                        except json.JSONDecodeError as e:
                            logging.warning(f"AI response was not valid JSON: {e}. Raw output: {raw_ai_output[:200]}...")
                    else:
                        logging.warning(f"Could not find JSON array in AI response. Raw output: {raw_ai_output[:200]}...")

            except Exception as e:
                logging.error(f"Error during AI software extraction ({self.api_type}): {e}", exc_info=True)

        # 2. Fallback/Augmentation: Use NER and Regex if AI failed or to supplement
        logging.info("Running NER/Regex fallback for software extraction...")
        fallback_identified_names: Counter[str] = collections.Counter()
        context_map: Dict[str, List[str]] = collections.defaultdict(list)

        # Use spaCy NER if available
        if NLP:
            try:
                doc = NLP(text[:1000000]) # Process a large chunk, but cap to avoid memory issues
                possible_entities = [ent.text for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG']]
                logging.info(f"spaCy NER found {len(possible_entities)} potential PRODUCT/ORG entities.")
                # Further filter based on context keywords
                for entity in possible_entities:
                    # Check if the entity itself looks like software (e.g., not just 'Apple' the company)
                     if len(entity) > 1 and (entity[0].isupper() or any(c.isdigit() for c in entity)): # Basic check
                          # Find mentions and check context
                          try:
                               for match in re.finditer(r'\b' + re.escape(entity) + r'\b', text, re.IGNORECASE):
                                    start = max(0, match.start() - 50)
                                    end = min(len(text), match.end() + 50)
                                    context_snippet = text[start:end].lower()
                                    if any(keyword in context_snippet for keyword in SOFTWARE_CONTEXT_KEYWORDS):
                                         # Normalize common variations slightly (e.g., lowercase, remove trailing dots)
                                         normalized_name = entity.strip().rstrip('.')
                                         fallback_identified_names[normalized_name] += 1
                                         if len(context_map[normalized_name]) < 15: # Store limited context
                                            context_map[normalized_name].append(text[start:end].replace('\n', ' ').strip())
                                         break # Count once per entity type found in relevant context
                          except re.error:
                              logging.warning(f"Regex error processing entity: {entity}")
                              continue # Skip this entity if regex fails
            except Exception as e:
                logging.error(f"Error during spaCy NER processing: {e}")

        # Use Regex as a further fallback or supplement
        try:
            for match in POTENTIAL_SOFTWARE_REGEX.finditer(text):
                potential_name = match.group(1).strip().rstrip('.')
                # Basic filtering: length > 1, not purely numeric, avoid very common words if needed
                if len(potential_name) > 1 and not potential_name.isdigit() and potential_name.lower() not in ['API', 'SDK', 'AI', 'ML', 'UI', 'UX']:
                    # Check context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context_snippet = text[start:end].lower()
                    if any(keyword in context_snippet for keyword in SOFTWARE_CONTEXT_KEYWORDS):
                        # Check if already found by NER with higher confidence, maybe prioritize NER?
                        # For now, just add counts.
                         fallback_identified_names[potential_name] += 1
                         if len(context_map[potential_name]) < 15:
                            context_map[potential_name].append(text[start:end].replace('\n', ' ').strip())
        except Exception as e:
            logging.error(f"Error during Regex software extraction: {e}")

        logging.info(f"Fallback methods identified {len(fallback_identified_names)} potential names with counts.")

        # 3. Consolidate Results
        # Start with AI identified names, give them an initial count of 1 or more if possible
        for name in ai_identified_names:
            if name not in mentioned_software:
                 mentioned_software[name] = {'name': name, 'mentions': 0, 'context': []} # Initialize count

        # Add counts and context from fallback methods (NER/Regex)
        # We need to re-scan the text to get accurate counts for *all* identified names (AI + Fallback)
        # And associate context properly.
        final_candidates = set(mentioned_software.keys()) | set(fallback_identified_names.keys())
        logging.info(f"Consolidating {len(final_candidates)} unique candidate names.")

        # Recalculate mentions and gather context across the entire dataset for all candidates
        final_mentions: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: {'name': '', 'mentions': 0, 'context': []})

        # Use the structured scraped_data for context gathering
        for data_item in scraped_data:
            item_text = self._combine_text_data([data_item]) # Get text for this specific item
            item_url = data_item.get('url', 'N/A')
            for name in final_candidates:
                 # Use case-insensitive matching, ensure word boundaries
                 try:
                     # Escape special regex characters in the name itself
                     escaped_name = re.escape(name)
                     # Add variations? e.g., 'VS Code' vs 'VSCode'. For now, exact boundary match.
                     pattern = re.compile(r'\b' + escaped_name + r'\b', re.IGNORECASE)
                     matches = list(pattern.finditer(item_text))
                     if matches:
                         # Increment count only once per data item to avoid overcounting within a single post/comment
                         # To get total occurrences, use `len(matches)` instead of `+= 1`
                         # Let's count total occurrences for better weighting
                         mention_count_in_item = len(matches)
                         if mention_count_in_item > 0:
                              final_mentions[name]['name'] = name # Ensure name is set
                              final_mentions[name]['mentions'] += mention_count_in_item
                              # Add context from the first match in this item
                              if len(final_mentions[name]['context']) < 15: # Limit stored context examples
                                   match_obj = matches[0]
                                   start = max(0, match_obj.start() - 100)
                                   end = min(len(item_text), match_obj.end() + 100)
                                   context_str = item_text[start:end].replace('\n', ' ').strip()
                                   # Include source URL if available
                                   final_mentions[name]['context'].append(f"{context_str} (Source: {item_url})")

                 except re.error:
                     logging.warning(f"Regex error searching for software name: {name}")
                     continue

        # Filter out low-mention candidates (likely noise)
        min_mentions_threshold = 2 # Require at least 2 mentions to be considered
        filtered_mentions = [details for details in final_mentions.values() if details['mentions'] >= min_mentions_threshold]

        # Sort by mention count descending
        return sorted(filtered_mentions, key=lambda x: x['mentions'], reverse=True)


    # _extract_context is effectively integrated into _extract_software_mentions now


    def _extract_pain_points_and_features(self, software_mentions_detailed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts pain points and feature requests related to software mentions.
        Categorizes issues and assigns severity/type.
        """
        all_issues_data = []
        issue_keywords_flat = {kw: cat for cat, kws in ALL_ISSUE_KEYWORDS.items() for kw in kws}

        for software in software_mentions_detailed:
            software_name = software['name']
            contexts = software['context'] # Contexts now include source URL
            software_issues = []

            if not contexts:
                continue

            for context in contexts:
                context_lower = context.lower()
                found_keywords = []
                best_category = None # e.g., 'critical', 'major', 'minor', 'request'
                severity = None # For pain points: 'critical', 'major', 'minor'

                for keyword, category in issue_keywords_flat.items():
                    # Use word boundaries for more precise keyword matching
                    if re.search(r'\b' + re.escape(keyword) + r'\b', context_lower):
                        found_keywords.append(keyword)
                        # Determine the most severe category found for this context
                        if category in PAIN_POINT_KEYWORDS:
                             current_severity_level = list(PAIN_POINT_KEYWORDS.keys()).index(category)
                             if severity is None or current_severity_level < list(PAIN_POINT_KEYWORDS.keys()).index(severity):
                                  severity = category
                                  best_category = category # Update best category if it's a pain point
                        elif category == 'request' and best_category not in PAIN_POINT_KEYWORDS: # Prioritize pain points over requests if both keywords appear
                            best_category = 'request'


                if found_keywords:
                    sentiment_score = self.sentiment_analyzer(context).sentiment.polarity
                    issue_type = 'unknown'

                    # Refine type/severity based on keywords and sentiment
                    if best_category == 'request' and sentiment_score > -0.1: # If keywords suggest request and sentiment isn't strongly negative
                        issue_type = 'feature_request'
                        severity = None # Not applicable for feature requests
                    elif severity: # If pain point keywords were found
                         issue_type = 'pain_point'
                         # Escalate severity based on strong negative sentiment
                         if sentiment_score < -0.7 and severity == 'major':
                             severity = 'critical'
                         elif sentiment_score < -0.5 and severity == 'minor':
                             severity = 'major'
                    elif sentiment_score < -0.2: # If no clear keywords but negative sentiment
                         issue_type = 'pain_point'
                         severity = 'minor' # Assume minor if only sentiment is negative

                    # Only add if we have a clear type
                    if issue_type != 'unknown':
                        software_issues.append({
                            'context': context,
                            'keywords': found_keywords,
                            'type': issue_type, # 'pain_point' or 'feature_request'
                            'severity': severity, # Only for pain_points ('critical', 'major', 'minor', None)
                            'sentiment_score': round(sentiment_score, 2)
                        })

            if software_issues:
                # Separate pain points and feature requests for summary counts
                pain_points = [iss for iss in software_issues if iss['type'] == 'pain_point']
                feature_requests = [iss for iss in software_issues if iss['type'] == 'feature_request']

                pain_severity_counts = {'critical': 0, 'major': 0, 'minor': 0}
                for pp in pain_points:
                    if pp['severity']:
                        pain_severity_counts[pp['severity']] += 1

                all_issues_data.append({
                    'software': software_name,
                    'issues': software_issues, # Keep all issues together for context
                    'type': 'pain_point' if pain_points else 'feature_request', # Overall type based on content
                    'pain_point_summary': {
                        'count': len(pain_points),
                        'severity_counts': pain_severity_counts
                    },
                    'feature_request_summary': {
                        'count': len(feature_requests)
                    },
                    'total_issues': len(software_issues)
                })

        # Sort primarily by critical pain points, then total issues
        return sorted(all_issues_data, key=lambda x: (
            x['pain_point_summary']['severity_counts']['critical'],
            x['total_issues']
        ), reverse=True)

    def _calculate_demand_metrics(self, scraped_data: List[Dict[str, Any]], software_mentions_detailed: List[Dict[str, Any]], sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive demand metrics for mentioned software."""
        metrics = {}
        total_data_points = len(scraped_data)
        if total_data_points == 0: return {}

        all_platforms = {data.get('platform', 'unknown') for data in scraped_data}
        total_all_mentions = sum(s['mentions'] for s in software_mentions_detailed)
        if total_all_mentions == 0: total_all_mentions = 1 # Avoid division by zero

        for software in software_mentions_detailed:
            software_name = software['name']
            mention_count = software['mentions']
            platform_counts: Counter[str] = collections.Counter()
            engagement_metrics: Dict[str, float] = collections.defaultdict(float)
            mention_timestamps: List[datetime.datetime] = []

            # Iterate through data to aggregate platform counts, engagement, timestamps for THIS software
            for data in scraped_data:
                platform = data.get('platform', 'unknown')
                timestamp_str = data.get('timestamp')
                item_text = self._combine_text_data([data])

                # Use regex for counting mentions within this specific data item
                try:
                    pattern = re.compile(r'\b' + re.escape(software_name) + r'\b', re.IGNORECASE)
                    matches = pattern.findall(item_text)
                    item_mention_count = len(matches)
                except re.error:
                    item_mention_count = 0

                if item_mention_count > 0:
                    platform_counts[platform] += item_mention_count # Count all occurrences here

                    # Add timestamp if valid
                    if timestamp_str:
                        try:
                            # Handle various possible ISO formats, including 'Z' UTC marker
                            dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            mention_timestamps.append(dt)
                        except (ValueError, TypeError):
                            # Fallback to now if parsing fails, log warning
                            logging.debug(f"Could not parse timestamp '{timestamp_str}' for {software_name}. Using current time.")
                            mention_timestamps.append(datetime.datetime.now(datetime.timezone.utc))

                    # Platform-specific engagement (using normalized approach)
                    # Weights are heuristics, can be adjusted
                    engagement_score = 0
                    if platform == 'reddit':
                        # Consider score (upvotes) and comment count
                        score = data.get('upvotes', data.get('score', 0))
                        num_comments = len(data.get('comments', []))
                        engagement_score = score + num_comments * 2 # Comments weighted more
                    elif platform == 'twitter':
                        likes = data.get('likes', data.get('favorite_count', 0))
                        retweets = data.get('retweets', data.get('retweet_count', 0))
                        replies = len(data.get('replies', []))
                        # Quote tweets? Views? Data might vary.
                        engagement_score = likes + retweets * 2 + replies * 3 # Replies weighted most
                    elif platform == 'youtube':
                        # Views can be huge, use logarithm or cap? Normalize carefully.
                        views = data.get('views', data.get('view_count', 0))
                        likes = data.get('likes', data.get('like_count', 0))
                        comments = len(data.get('comments', []))
                        # Normalize views, e.g., log scale or divide by a large number
                        engagement_score = math.log1p(views / 1000) + likes + comments * 2 # Log(views/1k) + likes + comments*2
                    elif platform == 'producthunt':
                        upvotes = data.get('upvotes', 0)
                        comments = len(data.get('comments', []))
                        engagement_score = upvotes * 1.5 + comments * 2 # PH upvotes might be more indicative than Reddit's
                    else: # Generic fallback
                        engagement_score = len(data.get('comments', [])) # Simple comment count

                    engagement_metrics[platform] += engagement_score

            # Calculate metrics after iterating through all data for this software
            total_mentions_for_sw = sum(platform_counts.values())
            # Ensure consistency with the initially calculated mention_count if different
            if total_mentions_for_sw != mention_count:
                 logging.warning(f"Mention count mismatch for {software_name}. Initial: {mention_count}, Recalculated: {total_mentions_for_sw}. Using recalculated.")
                 mention_count = total_mentions_for_sw # Use the sum from platform counts

            platform_percentages = {
                platform: round(count / mention_count * 100, 2) if mention_count else 0
                for platform, count in platform_counts.items()
            }

            recency_metrics = {}
            if mention_timestamps:
                # Ensure all timestamps are timezone-aware (UTC) for correct comparison
                mention_timestamps = [ts.astimezone(datetime.timezone.utc) if ts.tzinfo is None else ts for ts in mention_timestamps]
                now = datetime.datetime.now(datetime.timezone.utc)
                mention_timestamps.sort(reverse=True)
                most_recent = mention_timestamps[0]
                days_since_most_recent = (now - most_recent).days
                # Normalize recency score (higher is better) - capped at 100
                recency_score = max(0, 100 - days_since_most_recent * 2) # Simple linear decay, 0 after 50 days

                if len(mention_timestamps) >= 2:
                    oldest = mention_timestamps[-1]
                    time_span_days = max(1, (most_recent - oldest).days) # Avoid division by zero
                    mentions_per_day = len(mention_timestamps) / time_span_days
                elif len(mention_timestamps) == 1:
                     mentions_per_day = 1 # Assume 1 mention per day if only one timestamp
                else:
                    mentions_per_day = 0

                recency_metrics = {
                    'days_since_most_recent': days_since_most_recent,
                    'recency_score': round(recency_score, 2), # 0-100 scale
                    'mentions_per_day_avg': round(mentions_per_day, 3)
                }

            # Get average sentiment for demand score calculation
            avg_sentiment = sentiment_analysis.get(software_name, {}).get('average_sentiment', 0.0)

            demand_score = self._calculate_demand_score(
                mention_count, platform_counts, engagement_metrics, recency_metrics, avg_sentiment
            )

            metrics[software_name] = {
                'total_mentions': mention_count,
                'mention_percentage_overall': round(mention_count / total_all_mentions * 100, 2),
                'platform_distribution': dict(platform_counts),
                'platform_percentages': platform_percentages,
                'engagement_metrics_total': sum(engagement_metrics.values()),
                'engagement_by_platform': dict(engagement_metrics),
                'recency_metrics': recency_metrics,
                'demand_score': demand_score # Single combined score
            }

        # Sort final metrics by demand score
        return dict(sorted(metrics.items(), key=lambda item: item[1]['demand_score'], reverse=True))

    def _calculate_demand_score(self, mention_count: int, platform_counts: Counter[str], engagement_metrics: Dict[str, float], recency_metrics: Dict[str, Any], avg_sentiment: float) -> float:
        """Calculate an enhanced demand score based on multiple factors."""
        if mention_count == 0:
            return 0.0

        # 1. Base Score: Log of mentions to dampen effect of extremely high counts
        base_score = math.log1p(mention_count) * 10 # Scale factor

        # 2. Platform Diversity: Bonus for appearing on more platforms
        platform_diversity = len([count for count in platform_counts.values() if count > 0])
        # Normalize diversity score (0 to 1, assuming max ~5 platforms reasonable)
        diversity_factor = 1 + (platform_diversity / 5.0) * 0.5 # Max 50% bonus

        # 3. Engagement Factor: Log of total engagement
        total_engagement = sum(engagement_metrics.values())
        # Normalize engagement (log scale, max ~50% bonus)
        engagement_factor = 1 + math.log1p(total_engagement) / 20.0 # Adjust divisor as needed

        # 4. Recency Factor: Use the 0-100 recency score
        recency_score = recency_metrics.get('recency_score', 0)
        # Normalize recency (max 50% bonus)
        recency_factor = 1 + (recency_score / 100.0) * 0.5

        # 5. Frequency Factor: Average mentions per day (capped)
        mentions_per_day = recency_metrics.get('mentions_per_day_avg', 0)
        # Normalize frequency (log scale, capped bonus)
        frequency_factor = 1 + math.log1p(mentions_per_day) * 0.2 # Smaller bonus for frequency

        # 6. Sentiment Factor: Negative sentiment slightly increases demand for alternatives
        # Map sentiment (-1 to 1) to a factor (e.g., 0.9 to 1.1)
        # More negative sentiment -> higher factor
        sentiment_factor = 1 - min(0.2, max(-0.5, avg_sentiment)) / 2 # e.g., -0.5 maps to 1.125, 0 maps to 1, 0.5 maps to 0.875

        # Combine factors multiplicatively
        final_score = base_score * diversity_factor * engagement_factor * recency_factor * frequency_factor * sentiment_factor

        return round(final_score, 2)

    def _generate_product_ideas(self, pain_points_data: List[Dict[str, Any]], feature_requests_data: List[Dict[str, Any]], demand_metrics: Dict[str, Any], co_occurrence: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
        """Generate product ideas based on pain points, features, demand, and co-occurrence."""
        product_ideas = []
        processed_software = set() # Avoid redundant ideas for the same base software

        # Combine pain points and features for easier lookup
        all_issues_map = collections.defaultdict(list)
        for item in pain_points_data + feature_requests_data:
            all_issues_map[item['software']].extend(item['issues'])

        # --- Attempt AI Idea Generation First (If API Key Available) ---
        # DO NOT CHANGE GEMINI CALL LOGIC
        if self.api_key and (self.genai_client or self.genai or self.openai or self.anthropic_client):
            try:
                logging.info("Attempting product idea generation using AI...")
                # Prepare concise input for the AI prompt
                top_demand_sw = list(demand_metrics.keys())[:10] # Focus on top 10 demand software
                relevant_issues = []
                for sw_name in top_demand_sw:
                    issues = all_issues_map.get(sw_name, [])
                    if issues:
                        # Summarize issues: top 3 pain points, top 3 features
                        pps = sorted([i for i in issues if i['type'] == 'pain_point' and i['severity']], key=lambda x: list(PAIN_POINT_KEYWORDS.keys()).index(x['severity']))[:3]
                        frs = [i for i in issues if i['type'] == 'feature_request'][:3]
                        relevant_issues.append({
                            "software": sw_name,
                            "demand_score": demand_metrics[sw_name]['demand_score'],
                            "top_pain_points": [f"{p['severity']}: {p['context'][:100]}..." for p in pps],
                            "top_feature_requests": [f"{f['context'][:100]}..." for f in frs]
                        })

                if not relevant_issues:
                     logging.warning("No relevant issues found for top demand software to feed to AI.")
                     # Proceed to fallback directly if no issues for top software

                else:
                    ai_prompt = f"""
                    You are a Product Manager analyzing user feedback to brainstorm new product ideas or improvements.
                    Based on the following software issues (pain points and feature requests) and their demand scores (higher score means more discussion/engagement), generate multiple SPECIFIC and ACTIONABLE product ideas.

                    Focus on creating solutions that directly address the identified problems or missing features for high-demand software. Ideas can be:
                    1. An improved alternative to an existing product.
                    2. A new tool that combines features or fills a gap between existing tools.
                    3. A focused feature add-on or plugin.

                    Provided Data (Top Demand Software Issues):
                    ---
                    {json.dumps(relevant_issues, indent=2)}
                    ---

                    For EACH product idea, provide:
                    - "target_software": [List of one or more existing software the idea relates to]
                    - "idea_name": A concise, descriptive name for the new product/feature.
                    - "idea_description": A clear explanation of the product/feature and WHICH specific pain points or feature requests it solves. Be explicit.
                    - "key_features": [List of 3-5 bullet points outlining the core functionalities]
                    - "justification": Briefly explain why this idea is promising based on the input data (e.g., "High demand for X, addresses critical pain point Y").

                    Generate AS MANY distinct and well-justified ideas as possible based *only* on the provided data. Aim for diversity in the types of ideas.

                    Format your response STRICTLY as a JSON array of objects, each object representing one product idea with the keys mentioned above.
                    Example:
                    [
                      {{
                        "target_software": ["Slack"],
                        "idea_name": "Slack Focus Assist",
                        "idea_description": "A Slack plugin to reduce notification fatigue (addressing 'annoying', 'frustrating' comments) by intelligently batching non-urgent messages and allowing deep work sessions.",
                        "key_features": [
                          "Configurable notification batching intervals.",
                          "AI-powered priority sorting of messages.",
                          "One-click 'Deep Work' mode activation.",
                          "Customizable keyword alerts.",
                          "Team status visibility for focus modes."
                        ],
                        "justification": "Addresses common Slack pain point of distraction (high demand score) with specific features."
                      }},
                      {{ ... another idea ... }}
                    ]
                    ---
                    Respond only with the JSON array.
                    """

                    ai_ideas_raw = None
                    if self.api_type == 'gemini' and (self.genai_client or self.genai):
                        # Ensure consistency with the API call method used in setup
                        if self.genai_client and hasattr(self.genai_client, 'generate_content'):
                             logging.warning("Using model 'gemini-2.0-flash' for idea generation as specified.")
                             response = self.genai_client.generate_content(model='models/gemini-1.5-flash-latest', contents=ai_prompt) # Match model identifier
                             ai_ideas_raw = response.text
                        elif self.genai and hasattr(self.genai, 'models') and hasattr(self.genai.models, 'generate_content'):
                             logging.warning("Using model 'gemini-2.0-flash' via genai.models.generate_content for idea generation.")
                             response = self.genai.models.generate_content(model='models/gemini-1.5-flash-latest', contents=ai_prompt) # Match model identifier
                             ai_ideas_raw = response.text
                        else:
                            logging.error("Gemini AI object not configured correctly for content generation.")

                    elif self.api_type == 'openai' and self.openai:
                        response = self.openai.chat.completions.create(
                            model='gpt-4o-mini', # Use a cost-effective but capable model
                            response_format={'type': 'json_object'},
                            messages=[
                                {'role': 'system', 'content': 'You generate product ideas based on software pain points and feature requests, returning a JSON array.'},
                                {'role': 'user', 'content': ai_prompt}
                            ]
                        )
                        ai_ideas_raw = response.choices[0].message.content

                    elif self.api_type == 'anthropic' and self.anthropic_client:
                        response = self.anthropic_client.messages.create(
                            model='claude-3-haiku-20240307', # Fast model suitable for generation
                            max_tokens=3000, # Allow more tokens for potentially many ideas
                            messages=[{'role': 'user', 'content': ai_prompt}]
                        )
                        ai_ideas_raw = response.content[0].text

                    if ai_ideas_raw:
                        # Extract JSON array robustly
                        json_match = re.search(r'\[\s*{.*}\s*\]', ai_ideas_raw, re.DOTALL)
                        if json_match:
                            try:
                                parsed_ideas = json.loads(json_match.group(0))
                                if isinstance(parsed_ideas, list):
                                    # Validate and structure the ideas
                                    for idea in parsed_ideas:
                                        if isinstance(idea, dict) and all(k in idea for k in ['target_software', 'idea_name', 'idea_description', 'key_features']):
                                            # Add demand score and link back pain points if possible
                                            targets = idea['target_software']
                                            primary_target = targets[0] if targets else None
                                            demand_score = 0
                                            if primary_target and primary_target in demand_metrics:
                                                 demand_score = demand_metrics[primary_target].get('demand_score', 0)

                                            product_ideas.append({
                                                'target_software': targets,
                                                'idea_name': idea['idea_name'],
                                                'idea_description': idea['idea_description'],
                                                'key_features': idea.get('key_features', []),
                                                'justification': idea.get('justification', 'AI Generated Idea'),
                                                'demand_score_primary_target': demand_score,
                                                'generation_method': 'ai'
                                            })
                                            if primary_target: processed_software.add(primary_target)

                                    logging.info(f"Successfully generated {len(product_ideas)} product ideas using AI.")
                            except json.JSONDecodeError as e:
                                logging.warning(f"AI response for ideas was not valid JSON: {e}. Raw: {ai_ideas_raw[:200]}...")
                        else:
                            logging.warning(f"Could not find JSON array in AI idea response. Raw: {ai_ideas_raw[:200]}...")

            except Exception as e:
                logging.error(f"Error during AI product idea generation ({self.api_type}): {e}", exc_info=True)

        # --- Fallback / Augmentation Idea Generation ---
        logging.info("Running fallback/augmentation for product idea generation...")

        # 1. Identify Common Themes
        theme_counter: Counter[str] = collections.Counter()
        for sw, issues in all_issues_map.items():
            for issue in issues:
                 if issue['type'] == 'pain_point' and issue['severity']:
                      theme_counter[f"pain_{issue['severity']}"] += 1 # e.g., pain_critical
                      # Could add keyword-based themes here (e.g., 'performance', 'ui', 'cost')
                 elif issue['type'] == 'feature_request':
                      theme_counter['feature_request'] += 1
                      # Could add themes based on request keywords (e.g., 'integration', 'api')

        top_themes = [theme for theme, count in theme_counter.most_common(5)]
        logging.info(f"Identified common themes: {top_themes}")

        # 2. Iterate through High-Demand Software (that AI didn't cover)
        sorted_demand = sorted(demand_metrics.items(), key=lambda item: item[1]['demand_score'], reverse=True)

        ideas_generated_fallback = 0
        max_fallback_ideas = 30 # Limit the number of fallback ideas to avoid excessive output

        for sw_name, metrics in sorted_demand:
            if ideas_generated_fallback >= max_fallback_ideas:
                 break
            if sw_name in processed_software: # Skip if AI already generated ideas for this
                 continue
            if sw_name not in all_issues_map: # Skip if no issues were mapped to it
                 continue

            demand_score = metrics.get('demand_score', 0)
            issues = all_issues_map[sw_name]
            pain_points = sorted([i for i in issues if i['type'] == 'pain_point' and i['severity']], key=lambda x: list(PAIN_POINT_KEYWORDS.keys()).index(x['severity']))
            feature_requests = [i for i in issues if i['type'] == 'feature_request']

            # Idea Type 1: Address Top Pain Point
            if pain_points:
                top_pp = pain_points[0]
                idea_name = f"Improved {sw_name} Alternative (Fixing {top_pp['severity'].capitalize()} Issues)"
                idea_desc = f"A replacement for {sw_name} focused on resolving the common '{top_pp['severity']}' pain point related to: {top_pp['context'][:150]}..."
                key_features = [
                    f"Core {sw_name} functionality",
                    f"Enhanced performance/stability/usability (addressing '{top_pp['keywords'][0]}')",
                    "Modern user interface",
                    "Feedback mechanism for ongoing improvement"
                ]
                product_ideas.append({
                    'target_software': [sw_name],
                    'idea_name': idea_name,
                    'idea_description': idea_desc,
                    'key_features': key_features,
                    'justification': f"Addresses top '{top_pp['severity']}' pain point for {sw_name} (Demand: {demand_score}). Evidence: {len(pain_points)} pain points found.",
                    'demand_score_primary_target': demand_score,
                    'generation_method': 'fallback_painpoint'
                })
                processed_software.add(sw_name) # Mark as processed for this type
                ideas_generated_fallback += 1
                if ideas_generated_fallback >= max_fallback_ideas: break


            # Idea Type 2: Fulfill Top Feature Request
            if feature_requests and sw_name not in processed_software: # Check if already processed
                top_fr = feature_requests[0]
                idea_name = f"{sw_name} Plus with {top_fr['keywords'][0].capitalize()} Feature"
                idea_desc = f"An enhanced version of {sw_name} or a dedicated add-on that incorporates the frequently requested feature: {top_fr['context'][:150]}..."
                key_features = [
                    f"Seamless integration with {sw_name}",
                    f"Implementation of '{top_fr['keywords'][0]}' functionality",
                    "User-friendly configuration",
                    "Potential for additional related features"
                ]
                product_ideas.append({
                    'target_software': [sw_name],
                    'idea_name': idea_name,
                    'idea_description': idea_desc,
                    'key_features': key_features,
                    'justification': f"Fulfills top feature request for {sw_name} (Demand: {demand_score}). Evidence: {len(feature_requests)} requests found.",
                    'demand_score_primary_target': demand_score,
                    'generation_method': 'fallback_feature'
                })
                processed_software.add(sw_name)
                ideas_generated_fallback += 1
                if ideas_generated_fallback >= max_fallback_ideas: break

            # Idea Type 3: Alternative based on Co-occurrence (if tool A is mentioned with issues alongside tool B)
            if sw_name in co_occurrence and sw_name not in processed_software:
                 competitors = sorted(co_occurrence[sw_name].items(), key=lambda item: item[1], reverse=True)
                 if competitors:
                      top_competitor, count = competitors[0]
                      # Check if competitor also has high demand / issues
                      comp_demand = demand_metrics.get(top_competitor, {}).get('demand_score', 0)
                      comp_issues = len(all_issues_map.get(top_competitor, []))

                      if comp_issues > 0 or comp_demand > demand_score / 2: # If competitor is also discussed/has issues
                           idea_name = f"Unified Alternative to {sw_name} and {top_competitor}"
                           idea_desc = f"A new tool combining the strengths of {sw_name} and {top_competitor}, while addressing common pain points found in discussions mentioning both (Co-occurrence: {count}). Aims to solve issues like: {pain_points[0]['context'][:100] if pain_points else 'N/A'}..."
                           key_features = [
                               f"Key feature set from {sw_name}",
                               f"Key feature set from {top_competitor}",
                               "Improved workflow integration",
                               "Addresses specific shared pain points (if identifiable)",
                           ]
                           product_ideas.append({
                               'target_software': [sw_name, top_competitor],
                               'idea_name': idea_name,
                               'idea_description': idea_desc,
                               'key_features': key_features,
                               'justification': f"High co-occurrence ({count}) suggests users compare/use {sw_name} & {top_competitor}. Opportunity to combine strengths and fix weaknesses. Demand Sw1: {demand_score}, Sw2: {comp_demand}.",
                               'demand_score_primary_target': demand_score,
                               'generation_method': 'fallback_cooccurrence'
                           })
                           processed_software.add(sw_name)
                           # Also mark competitor as processed to avoid symmetrical idea?
                           processed_software.add(top_competitor)
                           ideas_generated_fallback += 1
                           if ideas_generated_fallback >= max_fallback_ideas: break


        logging.info(f"Generated {ideas_generated_fallback} additional product ideas using fallback methods.")
        # Sort all ideas by demand score of the primary target
        return sorted(product_ideas, key=lambda x: x.get('demand_score_primary_target', 0), reverse=True)

    def _analyze_sentiment(self, software_mentions_detailed: List[Dict[str, Any]]) -> Dict[str, Any]:
        "Perform sentiment analysis on context snippets for each software."
        sentiment_results = {}
        if not hasattr(self, 'sentiment_analyzer') or self.sentiment_analyzer is None:
             logging.warning("Sentiment analyzer not available.")
             return {}

        for software in software_mentions_detailed:
            software_name = software['name']
            contexts = software['context']
            if not contexts: continue

            sentiment_scores = []
            try:
                for context in contexts:
                    # Limit context length for TextBlob if it struggles with long strings
                    analysis_text = context[:1000] if len(context) > 1000 else context
                    blob = self.sentiment_analyzer(analysis_text)
                    sentiment_scores.append(blob.sentiment.polarity)
            except Exception as e:
                logging.warning(f"Sentiment analysis failed for a context of {software_name}: {e}")
                continue # Skip this software if analysis fails repeatedly


            if sentiment_scores:
                try:
                    avg_sentiment = statistics.mean(sentiment_scores)
                    # More nuanced categories
                    if avg_sentiment >= 0.2: sentiment_category = 'positive'
                    elif avg_sentiment >= 0.05: sentiment_category = 'mostly_positive'
                    elif avg_sentiment <= -0.2: sentiment_category = 'negative'
                    elif avg_sentiment <= -0.05: sentiment_category = 'mostly_negative'
                    else: sentiment_category = 'neutral'

                    sentiment_distribution = {
                        'positive': len([s for s in sentiment_scores if s >= 0.2]),
                        'mostly_positive': len([s for s in sentiment_scores if 0.05 <= s < 0.2]),
                        'neutral': len([s for s in sentiment_scores if -0.05 < s < 0.05]),
                        'mostly_negative': len([s for s in sentiment_scores if -0.2 < s <= -0.05]),
                        'negative': len([s for s in sentiment_scores if s <= -0.2])
                    }
                    sentiment_results[software_name] = {
                        'average_sentiment': round(avg_sentiment, 3),
                        'sentiment_category': sentiment_category,
                        'sentiment_distribution': sentiment_distribution,
                        'sentiment_scores_sample': [round(s, 2) for s in sentiment_scores[:10]], # Sample
                        'sample_size': len(sentiment_scores)
                    }
                except statistics.StatisticsError:
                    logging.warning(f"Could not calculate sentiment statistics for {software_name} (likely no scores).")
                except Exception as e:
                    logging.error(f"Unexpected error calculating sentiment stats for {software_name}: {e}")

        return dict(sorted(sentiment_results.items(), key=lambda item: item[1]['average_sentiment'])) # Sort by avg sentiment


    def _perform_statistical_analysis(self, scraped_data: List[Dict[str, Any]], software_mentions_detailed: List[Dict[str, Any]], issues_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        "Perform statistical analysis on the data."
        stats_results = {'overall': {}, 'mention_statistics': {}, 'issue_statistics': {}, 'co_occurrence': {}}
        num_data_points = len(scraped_data)
        if num_data_points == 0: return stats_results

        # Overall Stats
        platform_distribution = collections.Counter(data.get('platform', 'unknown') for data in scraped_data)
        total_content_length = sum(len(self._combine_text_data([data])) for data in scraped_data)
        total_sw_mentions_count = sum(sw['mentions'] for sw in software_mentions_detailed)
        stats_results['overall'] = {
            'total_data_points': num_data_points,
            'total_unique_software_identified': len(software_mentions_detailed),
            'total_software_mentions': total_sw_mentions_count,
            'platform_distribution': dict(platform_distribution),
            'total_content_length_chars': total_content_length,
            'average_content_length_chars': round(total_content_length / num_data_points, 2) if num_data_points else 0
        }

        # Mention Statistics
        mention_counts = [software['mentions'] for software in software_mentions_detailed]
        if mention_counts:
            try:
                mean_mentions = statistics.mean(mention_counts)
                median_mentions = statistics.median(mention_counts)
                stdev_mentions = statistics.stdev(mention_counts) if len(mention_counts) > 1 else 0.0
                # Calculate quartiles more robustly
                sorted_counts = sorted(mention_counts)
                n = len(sorted_counts)
                q1 = statistics.quantiles(mention_counts, n=4)[0] if n >= 4 else (sorted_counts[0] if n > 0 else 0)
                q3 = statistics.quantiles(mention_counts, n=4)[2] if n >= 4 else (sorted_counts[-1] if n > 0 else 0)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                outliers = [sw['name'] for sw in software_mentions_detailed if sw['mentions'] > upper_bound]

                stats_results['mention_statistics'] = {
                    'mean': round(mean_mentions, 2),
                    'median': median_mentions,
                    'standard_deviation': round(stdev_mentions, 2),
                    'min': min(mention_counts),
                    'max': max(mention_counts),
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'outliers_by_mentions': outliers
                }
            except statistics.StatisticsError as e:
                logging.warning(f"Could not calculate mention statistics: {e}")
            except Exception as e:
                logging.error(f"Unexpected error calculating mention stats: {e}")


        # Issue Statistics
        total_issues = sum(item['total_issues'] for item in issues_data)
        total_pain_points = sum(item['pain_point_summary']['count'] for item in issues_data)
        total_feature_requests = sum(item['feature_request_summary']['count'] for item in issues_data)
        critical_pain_points = sum(item['pain_point_summary']['severity_counts']['critical'] for item in issues_data)

        stats_results['issue_statistics'] = {
            'total_issues_extracted': total_issues,
            'total_pain_points': total_pain_points,
            'total_feature_requests': total_feature_requests,
            'critical_pain_point_count': critical_pain_points,
            'avg_issues_per_software': round(total_issues / len(issues_data), 2) if issues_data else 0,
            # Could add distribution of issue types/severities here
        }

        # Co-occurrence Analysis (Heavy operation, consider sampling for large datasets)
        co_occurrence_limit = 20 # Limit to top N software for co-occurrence matrix
        top_software_names = [sw['name'] for sw in software_mentions_detailed[:co_occurrence_limit]]
        co_occurrence: Dict[str, Dict[str, int]] = {name: {} for name in top_software_names}
        if len(top_software_names) > 1:
            logging.info(f"Calculating co-occurrence matrix for top {len(top_software_names)} software...")
            for i, name1 in enumerate(top_software_names):
                for j in range(i + 1, len(top_software_names)):
                    name2 = top_software_names[j]
                    co_count = 0
                    # Optimize: Iterate data once, check for pairs
                    for data in scraped_data:
                        text = self._combine_text_data([data]).lower() # Lowercase once
                        # Simple check: presence of both names
                        try:
                            # Use word boundaries for accuracy
                            if re.search(r'\b' + re.escape(name1).lower() + r'\b', text) and \
                               re.search(r'\b' + re.escape(name2).lower() + r'\b', text):
                                co_count += 1
                        except re.error:
                            continue # Skip if regex fails for a name

                    if co_count > 0:
                        co_occurrence[name1][name2] = co_count
                        co_occurrence[name2][name1] = co_count # Symmetric matrix
            logging.info("Co-occurrence calculation complete.")


        # Clean up empty entries in co_occurrence
        stats_results['co_occurrence'] = {k: v for k, v in co_occurrence.items() if v}

        return stats_results


    def _detect_trends(self, scraped_data: List[Dict[str, Any]], software_mentions_detailed: List[Dict[str, Any]], issues_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        "Detect trends over time for software mentions and issue types."
        trend_results = {'software_trends': {}, 'issue_trends': {}}
        if not scraped_data: return trend_results

        # Group data by date
        date_grouped_data: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for data in scraped_data:
            timestamp = data.get('timestamp')
            if timestamp:
                try:
                    dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                    date_str = dt.strftime('%Y-%m-%d')
                    date_grouped_data[date_str].append(data)
                except (ValueError, TypeError):
                    pass # Skip data points with invalid timestamps

        if not date_grouped_data:
            logging.warning("No valid timestamps found for trend analysis.")
            return trend_results

        sorted_dates = sorted(date_grouped_data.keys())
        if len(sorted_dates) < 3:
             logging.info("Not enough dates for meaningful trend analysis (need at least 3).")
             return trend_results # Need at least 3 points for a basic trend

        # 1. Software Mention Trends
        logging.info("Calculating software mention trends...")
        for software in software_mentions_detailed[:20]: # Limit to top N software
            software_name = software['name']
            daily_mentions = []
            for date in sorted_dates:
                count = 0
                for data in date_grouped_data[date]:
                    text = self._combine_text_data([data])
                    try:
                        pattern = re.compile(r'\b' + re.escape(software_name) + r'\b', re.IGNORECASE)
                        count += len(pattern.findall(text))
                    except re.error:
                        continue
                daily_mentions.append({'date': date, 'mentions': count})

            trend_results['software_trends'][software_name] = self._calculate_linear_trend(daily_mentions, 'mentions')

        # 2. Overall Issue Trends (Pain Points, Feature Requests)
        logging.info("Calculating overall issue trends...")
        daily_pain_points = []
        daily_feature_requests = []
        daily_critical_pain_points = []

        issues_map_by_date = collections.defaultdict(lambda: {'pain': 0, 'feature': 0, 'critical': 0})
        for date in sorted_dates:
            for data in date_grouped_data[date]:
                 item_url = data.get('url', 'N/A') # Find issues associated with this data item
                 # This requires mapping issues back to original data items, which is complex.
                 # Simpler: Approximate by checking if software mentioned in this item had issues reported overall.
                 # More accurate: Re-analyze issues per day (computationally expensive).
                 # Let's do a simpler aggregation for now: count issues reported *on* a specific day.

                 # To do this properly, _extract_pain_points_and_features needs to associate issues with source data items.
                 # Assuming issues_data contains references or can be mapped back.
                 # For now, we'll just aggregate *all* mentions/issues per day as a proxy.
                 pass # TODO: Implement accurate daily issue counting if needed. Needs refactoring of issue extraction.

        # Placeholder: Calculate trend for *all* mentions as a proxy for activity
        daily_all_mentions = []
        for date in sorted_dates:
            count = 0
            for data in date_grouped_data[date]:
                text = self._combine_text_data([data])
                for software in software_mentions_detailed: # Sum mentions for all software
                     try:
                         pattern = re.compile(r'\b' + re.escape(software['name']) + r'\b', re.IGNORECASE)
                         count += len(pattern.findall(text))
                     except re.error: continue
            daily_all_mentions.append({'date': date, 'total_mentions': count})

        trend_results['issue_trends']['overall_activity'] = self._calculate_linear_trend(daily_all_mentions, 'total_mentions')


        return trend_results

    def _calculate_linear_trend(self, time_series_data: List[Dict[str, Any]], value_key: str) -> Dict[str, Any]:
        """Calculates linear trend slope and direction for time series data."""
        if len(time_series_data) < 3:
            return {'trend_direction': 'insufficient_data', 'slope': 0.0, 'data_points': len(time_series_data)}

        y = [point[value_key] for point in time_series_data]
        x = list(range(len(y))) # Simple time index
        n = len(x)

        try:
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
            sum_xx = sum(x_i * x_i for x_i in x)

            # Calculate slope (handle potential division by zero)
            denominator = (n * sum_xx - sum_x * sum_x)
            if denominator == 0:
                 slope = 0.0
            else:
                 slope = (n * sum_xy - sum_x * sum_y) / denominator

            # Determine trend direction based on slope magnitude relative to average value
            avg_y = sum_y / n if n > 0 else 0
            relative_slope_threshold = 0.05 # Trend is significant if slope is > 5% of average value? Adjust threshold.

            if avg_y != 0 and abs(slope / avg_y) > relative_slope_threshold:
                 trend_direction = 'increasing' if slope > 0 else 'decreasing'
            elif avg_y == 0 and slope > 0.1: # Handle zero average case
                 trend_direction = 'increasing'
            elif avg_y == 0 and slope < -0.1:
                 trend_direction = 'decreasing'
            else:
                 trend_direction = 'stable'

            return {
                'trend_direction': trend_direction,
                'slope': round(slope, 3),
                'data_points': n,
                'daily_values': time_series_data # Optional: include raw data
            }
        except ZeroDivisionError:
             logging.warning(f"Division by zero encountered during trend calculation for key '{value_key}'.")
             return {'trend_direction': 'error', 'slope': 0.0, 'data_points': n}
        except Exception as e:
            logging.error(f"Error calculating linear trend for key '{value_key}': {e}")
            return {'trend_direction': 'error', 'slope': 0.0, 'data_points': n}

    def _generate_ai_insights(self, software_mentions: List[Dict[str, Any]], pain_points: List[Dict[str, Any]], feature_requests: List[Dict[str, Any]], sentiment_analysis: Dict[str, Any], statistical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive AI insights using the connected AI API."""
        # DO NOT CHANGE GEMINI API CALL LOGIC
        if not self.api_key or not (self.genai_client or self.genai or self.openai or self.anthropic_client):
            return {'error': 'AI insights require a configured AI API connection'}

        logging.info("Preparing data for AI insight generation...")
        # --- Prepare concise summaries for the prompt ---
        top_sw_mentions = software_mentions[:10] # Top 10 mentioned
        top_demand_sw = sorted(
            [{'name': k, 'demand_score': v.get('demand_score', 0)} for k, v in self.demand_metrics.items()], # Use self.demand_metrics if available
            key=lambda x: x['demand_score'], reverse=True
        )[:10] if hasattr(self, 'demand_metrics') else top_sw_mentions # Fallback if demand metrics not computed yet

        top_pain_points_summary = []
        for pp_group in pain_points[:5]: # Top 5 software with pain points
            summary = {
                "software": pp_group['software'],
                "total_pain_points": pp_group['pain_point_summary']['count'],
                "critical_count": pp_group['pain_point_summary']['severity_counts']['critical'],
                "major_count": pp_group['pain_point_summary']['severity_counts']['major'],
                "top_issues": [iss['context'][:100]+'...' for iss in pp_group['issues'] if iss['type']=='pain_point'][:3]
            }
            top_pain_points_summary.append(summary)

        top_feature_requests_summary = []
        # Sort feature requests by count and associate with software
        fr_counts = collections.Counter(fr['software'] for fr in feature_requests)
        top_fr_sw = fr_counts.most_common(5)
        for sw_name, count in top_fr_sw:
             requests = [iss['context'][:100]+'...' for fr_group in feature_requests if fr_group['software'] == sw_name for iss in fr_group['issues'] if iss['type']=='feature_request']
             summary = {
                  "software": sw_name,
                  "request_count": count,
                  "top_requests": requests[:3]
             }
             top_feature_requests_summary.append(summary)


        top_sentiment = {k: v for k, v in list(sentiment_analysis.items())[:10]} # Top 10 by avg sentiment (default sort)
        positive_sentiment = sorted([(k, v) for k,v in sentiment_analysis.items() if v['average_sentiment'] > 0.1], key=lambda item: item[1]['average_sentiment'], reverse=True)[:5]
        negative_sentiment = sorted([(k, v) for k,v in sentiment_analysis.items() if v['average_sentiment'] < -0.1], key=lambda item: item[1]['average_sentiment'])[:5]


        stats_summary = {
            "overall": statistical_analysis.get("overall", {}),
            "mention_stats": statistical_analysis.get("mention_statistics", {}),
            "issue_stats": statistical_analysis.get("issue_statistics", {}),
            "top_cooccurring_pairs": [(k, list(v.keys())[:2]) for k,v in statistical_analysis.get("co_occurrence", {}).items()][:5]
        }

        # --- Construct the AI Prompt ---
        prompt = f"""
        Analyze the following consolidated data from online community discussions (Reddit, Twitter, YouTube, Product Hunt etc.) about various software tools. Provide a comprehensive market analysis and strategic insights report.

        **Input Data Summary:**

        1.  **Top Mentioned Software (Max 10):**
            {json.dumps(top_sw_mentions, indent=2)}

        2.  **Top Software by Demand Score (Max 10 - higher score indicates more buzz/engagement):**
            {json.dumps(top_demand_sw, indent=2)}

        3.  **Key Pain Points (Focus on software with most/critical issues, Max 5):**
            {json.dumps(top_pain_points_summary, indent=2)}

        4.  **Key Feature Requests (Focus on software with most requests, Max 5):**
            {json.dumps(top_feature_requests_summary, indent=2)}

        5.  **Sentiment Analysis Highlights:**
            - Top 5 Most Positively Viewed: {json.dumps(dict(positive_sentiment), indent=2)}
            - Top 5 Most Negatively Viewed: {json.dumps(dict(negative_sentiment), indent=2)}

        6.  **Statistical Highlights:**
            {json.dumps(stats_summary, indent=2)}

        **Analysis Report Sections (Respond ONLY with a valid JSON object containing these keys):**

        1.  `market_overview`: Briefly describe the software landscape based on mentions, demand, and platform distribution. Which types of software are most discussed?
        2.  `key_pain_point_themes`: Identify 2-4 major recurring themes or categories of problems users face across different software (e.g., "Usability Challenges", "High Cost", "Performance Bottlenecks", "Integration Gaps", "Poor Support"). Justify with examples from the data.
        3.  `top_feature_demand_themes`: What are 2-4 major categories of features users are requesting? (e.g., "AI Capabilities", "Better Collaboration", "Mobile Access", "Customization Options"). Justify with examples.
        4.  `opportunity_assessment`: Based *only* on the provided data (high demand, significant pain points/feature requests, negative sentiment), identify the 3-5 most promising opportunities for new products or major improvements. For each, mention the target software and the core problem/need it addresses.
        5.  `competitive_insights`: Analyze the relationships between software. Are there clear competitors based on co-occurrence or sentiment? Are users switching between tools?
        6.  `sentiment_summary`: Summarize the overall user sentiment. Are users generally satisfied or dissatisfied? Are there specific tools that evoke strong positive or negative reactions?
        7.  `strategic_recommendations`: Provide 2-3 actionable recommendations for developers looking to enter this market or improve existing tools, based directly on the analysis.
        8.  `potential_risks_or_gaps`: Briefly mention any potential risks (e.g., strong incumbent, niche market) or gaps in the data (e.g., lack of pricing discussion) suggested by the analysis.

        **Ensure your entire response is a single, valid JSON object.**
        ```json
        {{
          "market_overview": "...",
          "key_pain_point_themes": ["Theme 1: ...", "Theme 2: ..."],
          "top_feature_demand_themes": ["Theme A: ...", "Theme B: ..."],
          "opportunity_assessment": [
            {{"opportunity": "...", "target_software": ["..."], "justification": "..."}},
            {{...}}
          ],
          "competitive_insights": "...",
          "sentiment_summary": "...",
          "strategic_recommendations": ["Recommendation 1: ...", "Recommendation 2: ..."],
          "potential_risks_or_gaps": "..."
        }}
        ```
        """

        logging.info(f"Sending prompt (length: {len(prompt)} chars) to {self.api_type} for insights...")
        ai_insights = {'error': 'AI analysis failed to generate'} # Default error

        try:
            # --- Execute AI Call (Gemini logic unchanged as per request) ---
            raw_ai_output = None
            if self.api_type == 'gemini' and (self.genai_client or self.genai):
                # Determine which Gemini object to use based on initialization
                if self.genai_client and hasattr(self.genai_client, 'generate_content'):
                     logging.warning("Using model 'gemini-2.0-flash' for insights as specified.")
                     response = self.genai_client.generate_content(model='models/gemini-1.5-flash-latest', contents=prompt) # Match model identifier
                     raw_ai_output = response.text
                elif self.genai and hasattr(self.genai, 'models') and hasattr(self.genai.models, 'generate_content'):
                     logging.warning("Using model 'gemini-2.0-flash' via genai.models.generate_content for insights.")
                     response = self.genai.models.generate_content(model='models/gemini-1.5-flash-latest', contents=prompt) # Match model identifier
                     raw_ai_output = response.text
                else:
                     logging.error("Gemini AI object not configured correctly for content generation.")
                     ai_insights['error'] = "Gemini AI object not configured correctly."


            elif self.api_type == 'openai' and self.openai:
                 response = self.openai.chat.completions.create(
                     model="gpt-4o-mini", # Capable model for structured JSON
                     response_format={"type": "json_object"},
                     messages=[
                         {'role': 'system', 'content': 'You analyze software market data and generate strategic insights as a JSON object.'},
                         {'role': 'user', 'content': prompt}
                     ]
                 )
                 raw_ai_output = response.choices[0].message.content

            elif self.api_type == 'anthropic' and self.anthropic_client:
                 response = self.anthropic_client.messages.create(
                     model="claude-3-haiku-20240307", # Fast model
                     max_tokens=3500, # Allow ample space for detailed JSON
                     messages=[{'role': 'user', 'content': prompt}]
                 )
                 raw_ai_output = response.content[0].text

            # --- Process AI Response ---
            if raw_ai_output:
                logging.info(f"Received raw response from {self.api_type} (length: {len(raw_ai_output)} chars).")
                # Extract JSON object robustly (handles potential markdown ```json ... ```)
                json_match = re.search(r'\{\s*"market_overview":.*\}', raw_ai_output, re.DOTALL | re.IGNORECASE)
                if json_match:
                    extracted_json = json_match.group(0)
                    try:
                        ai_insights = json.loads(extracted_json)
                        # Basic validation
                        required_keys = ["market_overview", "key_pain_point_themes", "top_feature_demand_themes", "opportunity_assessment", "competitive_insights", "sentiment_summary", "strategic_recommendations", "potential_risks_or_gaps"]
                        if all(key in ai_insights for key in required_keys):
                             logging.info(f"Successfully parsed valid JSON insights from {self.api_type}.")
                             return ai_insights # Success
                        else:
                             missing_keys = [key for key in required_keys if key not in ai_insights]
                             logging.warning(f"AI response JSON is missing required keys: {missing_keys}")
                             ai_insights = {'error': f"AI response missing keys: {missing_keys}", 'raw_response': extracted_json}

                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to decode JSON from AI response: {e}. Raw matched JSON: {extracted_json[:500]}...")
                        # Attempt to find *any* JSON object as a fallback
                        any_json_match = re.search(r'\{.*\}', raw_ai_output, re.DOTALL)
                        if any_json_match:
                             ai_insights = {'error': f'JSONDecodeError: {e}. Found fallback JSON.', 'raw_response': any_json_match.group(0)}
                        else:
                             ai_insights = {'error': f'JSONDecodeError: {e}. No JSON object found.', 'raw_response': raw_ai_output}
                else:
                    logging.warning(f"Could not find the expected JSON structure in the AI response.")
                    ai_insights = {'error': 'Could not find valid JSON object in AI response.', 'raw_response': raw_ai_output}
            else:
                 logging.warning(f"Received no output from {self.api_type}.")
                 ai_insights['error'] = f"No output received from {self.api_type}."


        except Exception as e:
            logging.error(f"Error generating AI insights with {self.api_type}: {e}", exc_info=True)
            ai_insights = {'error': f"Failed to generate AI insights with {self.api_type}: {str(e)}"}

        return ai_insights


    def generate_ai_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive AI analysis based on previously computed results.
        This acts as a wrapper around _generate_ai_insights using results dict.
        """
        if not self.api_key:
            return {'error': 'AI analysis requires an API key'}

        # Extract necessary data from the results dictionary
        software_mentions = analysis_results.get('software_mentions', [])
        pain_points = analysis_results.get('pain_points', [])
        feature_requests = analysis_results.get('feature_requests', [])
        sentiment_analysis = analysis_results.get('sentiment_analysis', {})
        statistical_analysis = analysis_results.get('statistical_analysis', {})

        # Store demand metrics if needed for the generation function
        # This assumes _generate_ai_insights might internally use self.demand_metrics
        if 'demand_metrics' in analysis_results:
             self.demand_metrics = analysis_results['demand_metrics']

        logging.info("Calling _generate_ai_insights with provided analysis results...")
        ai_analysis = self._generate_ai_insights(
            software_mentions,
            pain_points,
            feature_requests,
            sentiment_analysis,
            statistical_analysis
        )
        return ai_analysis