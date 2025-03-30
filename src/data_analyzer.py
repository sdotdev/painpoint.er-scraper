#!/usr/bin/env python3
"""
Data Analyzer module for PainPoint.er Search
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

# For sentiment analysis
from textblob import TextBlob

# AI API clients
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from azure.ai.inference import ModelClient
    from azure.core.credentials import AzureKeyCredential
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False


class DataAnalyzer:
    """Analyze scraped data to identify software mentions, pain points, and opportunities."""
    
    def __init__(self, api_key: Optional[str] = None, api_type: str = 'gemini'):
        self.api_key = api_key
        self.api_type = api_type
        
        # Initialize AI client if API key is provided
        if api_key:
            self._initialize_ai_client()
    
    def _initialize_ai_client(self):
        """Initialize the appropriate AI client based on api_type."""
        if self.api_type == 'gemini' and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        elif self.api_type == 'openai' and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.api_type == 'anthropic' and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.api_type == 'deepseek' and DEEPSEEK_AVAILABLE:
            # DeepSeek uses GitHub token via Azure AI Inference
            self.client = ModelClient(
                endpoint="https://api.github.com/models/deepseek-ai/deepseek-v3",
                credential=AzureKeyCredential(self.api_key)
            )
    
    def analyze_data(self, scraped_data: List[Dict]) -> Dict:
        """Analyze scraped data to extract insights."""
        print("Analyzing scraped data...")
        
        # Extract software mentions
        software_mentions = self._extract_software_mentions(scraped_data)
        
        # Identify pain points
        pain_points = self._identify_pain_points(scraped_data, software_mentions)
        
        # Calculate demand metrics
        demand_metrics = self._calculate_demand_metrics(scraped_data, software_mentions)
        
        # Perform sentiment analysis
        sentiment_analysis = self._analyze_sentiment(scraped_data, software_mentions)
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(demand_metrics)
        
        # Generate product ideas based on pain points and demand metrics
        product_ideas = self._generate_product_ideas(pain_points, demand_metrics, sentiment_analysis)
        
        # Return combined analysis results
        return {
            'software_mentions': software_mentions,
            'pain_points': pain_points,
            'demand_metrics': demand_metrics,
            'sentiment_analysis': sentiment_analysis,
            'statistical_analysis': statistical_analysis,
            'product_ideas': product_ideas
        }
    
    def _extract_software_mentions(self, scraped_data: List[Dict]) -> List[Dict]:
        """Extract software mentions from scraped data."""
        software_mentions = []
        software_counter = Counter()
        
        # Common software-related terms to help identify mentions
        software_indicators = [
            'app', 'software', 'tool', 'platform', 'service', 'product',
            'library', 'framework', 'API', 'plugin', 'extension', 'add-on'
        ]
        
        for item in scraped_data:
            # Skip items with errors
            if 'error' in item:
                continue
            
            # Extract text content based on platform
            content = self._extract_text_content(item)
            
            # Use regex to find potential software mentions
            # Look for capitalized words or words followed by software indicators
            potential_mentions = set()
            
            # Find capitalized words that might be product names
            cap_pattern = r'\b[A-Z][a-zA-Z0-9]+(\s[A-Z][a-zA-Z0-9]+)?\b'
            cap_matches = re.findall(cap_pattern, content)
            potential_mentions.update(cap_matches)
            
            # Find words near software indicators
            for indicator in software_indicators:
                pattern = fr'\b([A-Za-z0-9]+)\s+{indicator}\b'
                matches = re.findall(pattern, content.lower())
                potential_mentions.update(matches)
            
            # Count mentions
            for mention in potential_mentions:
                if len(mention) > 2:  # Filter out very short mentions
                    software_counter[mention] += 1
        
        # Convert counter to list of dictionaries
        for software, count in software_counter.most_common(50):  # Limit to top 50
            software_mentions.append({
                'software': software,
                'mentions': count
            })
        
        return software_mentions
    
    def _identify_pain_points(self, scraped_data: List[Dict], software_mentions: List[Dict]) -> List[Dict]:
        """Identify pain points for each software mention."""
        pain_points = []
        
        # Pain point indicators
        pain_indicators = [
            'issue', 'problem', 'bug', 'hate', 'difficult', 'annoying',
            'frustrating', 'terrible', 'awful', 'wish', 'should', 'missing',
            'lacks', 'can\'t', 'cannot', 'doesn\'t', 'does not', 'broken',
            'useless', 'expensive', 'slow', 'crash', 'error', 'fail'
        ]
        
        # Process each software mention
        for mention in software_mentions:
            software = mention['software']
            software_pain_points = []
            
            # Look for pain points in each data item
            for item in scraped_data:
                # Skip items with errors
                if 'error' in item:
                    continue
                
                # Extract text content
                content = self._extract_text_content(item)
                
                # Skip if software is not mentioned
                if software.lower() not in content.lower():
                    continue
                
                # Split content into sentences
                sentences = re.split(r'[.!?]\s+', content)
                
                # Look for pain indicators in sentences that mention the software
                for sentence in sentences:
                    if software.lower() in sentence.lower():
                        for indicator in pain_indicators:
                            if indicator in sentence.lower():
                                # Calculate sentiment score for the sentence
                                sentiment = TextBlob(sentence).sentiment.polarity
                                
                                # Determine severity based on sentiment
                                severity = 'high' if sentiment < -0.5 else \
                                          'medium' if sentiment < -0.2 else 'low'
                                
                                # Add pain point
                                pain_point = {
                                    'context': sentence.strip(),
                                    'keywords': [indicator],
                                    'severity': severity,
                                    'sentiment_score': sentiment,
                                    'source': item.get('url', 'unknown')
                                }
                                
                                # Check if this is a duplicate
                                if not any(pp['context'] == pain_point['context'] for pp in software_pain_points):
                                    software_pain_points.append(pain_point)
                                break
            
            # Add to results if pain points were found
            if software_pain_points:
                pain_points.append({
                    'software': software,
                    'pain_points': software_pain_points
                })
        
        return pain_points
    
    def _calculate_demand_metrics(self, scraped_data: List[Dict], software_mentions: List[Dict]) -> Dict:
        """Calculate demand metrics for each software mention."""
        demand_metrics = {}
        
        for mention in software_mentions:
            software = mention['software']
            total_mentions = mention['mentions']
            
            # Initialize metrics
            metrics = {
                'total_mentions': total_mentions,
                'platform_distribution': defaultdict(int),
                'platform_percentages': {},
                'recency_metrics': {
                    'recent_mentions': 0,
                    'recency_score': 0
                },
                'demand_score': 0  # Will be calculated
            }
            
            # Count mentions by platform
            for item in scraped_data:
                if 'error' in item:
                    continue
                
                content = self._extract_text_content(item)
                platform = item.get('platform', 'unknown')
                
                if software.lower() in content.lower():
                    metrics['platform_distribution'][platform] += 1
            
            # Calculate platform percentages
            for platform, count in metrics['platform_distribution'].items():
                metrics['platform_percentages'][platform] = round((count / total_mentions) * 100, 2)
            
            # Calculate demand score (simple formula)
            # Higher score = more mentions across more platforms
            platform_diversity = len(metrics['platform_distribution'])
            metrics['demand_score'] = total_mentions * (1 + (platform_diversity / 10))
            
            demand_metrics[software] = metrics
        
        return demand_metrics
    
    def _analyze_sentiment(self, scraped_data: List[Dict], software_mentions: List[Dict]) -> Dict:
        """Perform sentiment analysis for each software mention."""
        sentiment_analysis = {}
        
        for mention in software_mentions:
            software = mention['software']
            sentiment_scores = []
            
            # Collect sentiment scores for all mentions
            for item in scraped_data:
                if 'error' in item:
                    continue
                
                content = self._extract_text_content(item)
                
                # Skip if software is not mentioned
                if software.lower() not in content.lower():
                    continue
                
                # Split content into sentences
                sentences = re.split(r'[.!?]\s+', content)
                
                # Analyze sentiment of sentences mentioning the software
                for sentence in sentences:
                    if software.lower() in sentence.lower():
                        sentiment = TextBlob(sentence).sentiment.polarity
                        sentiment_scores.append(sentiment)
            
            # Skip if no sentiment scores
            if not sentiment_scores:
                continue
            
            # Calculate average sentiment
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Categorize sentiment
            sentiment_category = 'positive' if average_sentiment > 0.2 else \
                               'negative' if average_sentiment < -0.2 else 'neutral'
            
            # Count sentiment distribution
            sentiment_distribution = {
                'positive': sum(1 for s in sentiment_scores if s > 0.2),
                'neutral': sum(1 for s in sentiment_scores if -0.2 <= s <= 0.2),
                'negative': sum(1 for s in sentiment_scores if s < -0.2)
            }
            
            sentiment_analysis[software] = {
                'average_sentiment': round(average_sentiment, 2),
                'sentiment_category': sentiment_category,
                'sentiment_distribution': sentiment_distribution,
                'sample_size': len(sentiment_scores)
            }
        
        return sentiment_analysis
    
    def _perform_statistical_analysis(self, demand_metrics: Dict) -> Dict:
        """Perform statistical analysis on demand metrics."""
        # Extract mention counts
        mention_counts = [metrics['total_mentions'] for _, metrics in demand_metrics.items()]
        
        if not mention_counts:
            return {'mention_statistics': {}}
        
        # Calculate statistics
        stats = {
            'mean': statistics.mean(mention_counts) if mention_counts else 0,
            'median': statistics.median(mention_counts) if mention_counts else 0,
            'standard_deviation': statistics.stdev(mention_counts) if len(mention_counts) > 1 else 0,
            'min': min(mention_counts) if mention_counts else 0,
            'max': max(mention_counts) if mention_counts else 0
        }
        
        return {'mention_statistics': stats}
    
    def _extract_text_content(self, item: Dict) -> str:
        """Extract text content from a scraped data item based on platform."""
        platform = item.get('platform', 'generic')
        content = ""
        
        if platform == 'reddit':
            content = f"{item.get('title', '')} {item.get('post_content', '')}"
            # Add comments
            for comment in item.get('comments', []):
                content += f" {comment.get('text', '')}"
        
        elif platform == 'twitter':
            content = item.get('tweet_content', '')
            # Add replies
            for reply in item.get('replies', []):
                content += f" {reply.get('text', '')}"
        
        elif platform == 'youtube':
            content = f"{item.get('title', '')} {item.get('description', '')}"
            # Add comments
            for comment in item.get('comments', []):
                content += f" {comment.get('text', '')}"
        
        elif platform == 'producthunt':
            content = f"{item.get('product_name', '')} {item.get('description', '')}"
            # Add comments
            for comment in item.get('comments', []):
                content += f" {comment.get('text', '')}"
        
        else:  # generic
            content = item.get('content', '')
        
        return content
    
    def generate_ai_analysis(self, analysis_results: Dict) -> Dict:
        """Generate AI analysis of the results."""
        if not self.api_key:
            return {"error": "No API key provided for AI analysis"}
        
        try:
            # Format the prompt with analysis results
            prompt = self._format_ai_prompt(analysis_results)
            
            # Call appropriate AI service based on api_type
            if self.api_type == 'gemini' and GEMINI_AVAILABLE:
                return self._call_gemini_api(prompt)
            elif self.api_type == 'openai' and OPENAI_AVAILABLE:
                return self._call_openai_api(prompt)
            elif self.api_type == 'anthropic' and ANTHROPIC_AVAILABLE:
                return self._call_anthropic_api(prompt)
            elif self.api_type == 'deepseek' and DEEPSEEK_AVAILABLE:
                return self._call_deepseek_api(prompt)
            else:
                return {"error": f"AI service {self.api_type} not available or not supported"}
        
        except Exception as e:
            return {"error": f"Error generating AI analysis: {str(e)}"}
    
    def _format_ai_prompt(self, analysis_results: Dict) -> str:
        """Format the prompt for AI analysis."""
        # Extract key information for the prompt
        software_mentions = analysis_results.get('software_mentions', [])
        pain_points = analysis_results.get('pain_points', [])
        demand_metrics = analysis_results.get('demand_metrics', {})
        sentiment_analysis = analysis_results.get('sentiment_analysis', {})
        
        # Format software mentions
        software_str = "\n".join([f"- {s['software']}: {s['mentions']} mentions" 
                              for s in software_mentions[:10]])  # Top 10 only
        
        # Format pain points (limited to top 3 per software)
        pain_points_str = ""
        for pp_group in pain_points:
            software = pp_group['software']
            pain_points_str += f"\n{software}:\n"
            for pp in pp_group['pain_points'][:3]:  # Top 3 only
                severity = pp.get('severity', 'unknown')
                pain_points_str += f"- [{severity}] {pp['context']}\n"
        
        # Format demand metrics
        demand_str = "\n".join([f"- {software}: score {metrics['demand_score']:.2f}" 
                            for software, metrics in sorted(demand_metrics.items(), 
                                                          key=lambda x: x[1]['demand_score'], 
                                                          reverse=True)[:10]])  # Top 10 only
        
        # Format sentiment analysis
        sentiment_str = "\n".join([f"- {software}: {data['sentiment_category']} ({data['average_sentiment']:.2f})" 
                               for software, data in sentiment_analysis.items()])  # Top 10 only
        
        # Construct the prompt
        prompt = f"""
        You are a product strategy consultant analyzing software pain points and market opportunities.
        
        Below is data from an analysis of online discussions about various software products.
        Please provide insights in the following areas:
        
        1. Top mentioned software products:
        {software_str}
        
        2. Key pain points identified:
        {pain_points_str}
        
        3. Demand metrics (higher score = more potential):
        {demand_str}
        
        4. Sentiment analysis:
        {sentiment_str}
        
        Based on this data, please provide:
        
        1. A summary of the most significant pain points across all software
        2. Identification of potential product opportunities
        3. Recommendations for addressing the top pain points
        4. Market trends that can be inferred from this data
        5. Suggestions for further research or data collection
        
        Format your response as a structured report with clear sections.
        """
        
        return prompt
    
    def _call_gemini_api(self, prompt: str) -> Dict:
        """Call Gemini API for analysis."""
        response = self.model.generate_content(prompt)
        
        # Parse the response into sections
        text = response.text
        sections = self._parse_ai_response(text)
        
        return sections
    
    def _call_openai_api(self, prompt: str) -> Dict:
        """Call OpenAI API for analysis."""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a product strategy consultant analyzing software pain points and market opportunities."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response into sections
        text = response.choices[0].message.content
        sections = self._parse_ai_response(text)
        
        return sections
    
    def _call_anthropic_api(self, prompt: str) -> Dict:
        """Call Anthropic API for analysis."""
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response into sections
        text = response.content[0].text
        sections = self._parse_ai_response(text)
        
        return sections
    
    def _call_deepseek_api(self, prompt: str) -> Dict:
        """Call DeepSeek API for analysis."""
        response = self.client.complete(
            messages=[
                {"role": "system", "content": "You are a product strategy consultant analyzing software pain points and market opportunities."},
                {"role": "user", "content": prompt}
            ],
            model="deepseek-ai/deepseek-v3"
        )
        
        # Parse the response into sections
        text = response.choices[0].message.content
        sections = self._parse_ai_response(text)
        
        return sections
    
    def _generate_product_ideas(self, pain_points: List[Dict], demand_metrics: Dict, sentiment_analysis: Dict) -> List[Dict]:
        """Generate product ideas based on pain points and demand metrics."""
        product_ideas = []
        
        # Process each software with pain points
        for pp_group in pain_points:
            software = pp_group['software']
            software_pain_points = pp_group['pain_points']
            
            # Skip if no pain points or not in demand metrics
            if not software_pain_points or software not in demand_metrics:
                continue
            
            # Get demand score
            demand_score = demand_metrics[software]['demand_score']
            
            # Group pain points by severity
            pain_by_severity = {'high': [], 'medium': [], 'low': []}
            for pp in software_pain_points:
                severity = pp.get('severity', 'low')
                pain_by_severity[severity].append(pp)
            
            # Generate ideas for high and medium severity pain points
            for severity in ['high', 'medium']:
                if not pain_by_severity[severity]:
                    continue
                
                # Take the top 3 pain points for this severity
                top_pains = pain_by_severity[severity][:3]
                
                # Extract contexts
                contexts = [pp['context'] for pp in top_pains]
                
                # Generate idea based on pain points
                idea_description = self._create_idea_description(software, contexts, severity)
                
                # Calculate idea score based on demand and severity
                severity_multiplier = 1.5 if severity == 'high' else 1.2
                idea_score = demand_score * severity_multiplier
                
                # Add sentiment adjustment if available
                if software in sentiment_analysis:
                    sentiment_score = sentiment_analysis[software].get('average_sentiment', 0)
                    # Negative sentiment increases idea value (more pain = more opportunity)
                    if sentiment_score < 0:
                        idea_score *= (1 + abs(sentiment_score))
                
                # Create product idea
                product_idea = {
                    'target_software': software,
                    'idea_description': idea_description,
                    'demand_score': idea_score,
                    'pain_points_addressed': contexts,
                    'severity': severity
                }
                
                product_ideas.append(product_idea)
        
        # Sort ideas by demand score (highest first)
        product_ideas.sort(key=lambda x: x['demand_score'], reverse=True)
        
        return product_ideas
    
    def _create_idea_description(self, software: str, pain_contexts: List[str], severity: str) -> str:
        """Create a product idea description based on pain points."""
        # Extract key issues from contexts
        issues = []
        for context in pain_contexts:
            # Extract a simplified version of the pain point
            simplified = re.sub(r'\b' + re.escape(software) + r'\b', 'it', context, flags=re.IGNORECASE)
            issues.append(simplified.strip())
        
        # Create a combined issue description
        if len(issues) == 1:
            issue_desc = f"Users complain that {issues[0]}"
        else:
            issue_desc = "Users complain that " + ", ".join(issues[:-1]) + f" and {issues[-1]}"
        
        # Generate idea based on severity
        if severity == 'high':
            return f"Develop an alternative to {software} that addresses critical issues: {issue_desc}."
        else:
            return f"Create a plugin or extension for {software} that solves: {issue_desc}."
    
    def _parse_ai_response(self, text: str) -> Dict:
        """Parse AI response into structured sections."""
        sections = {}
        
        # Look for section headers in the response
        section_patterns = [
            (r'(?i)(?:summary|overview)\s+of\s+(?:the\s+)?(?:most\s+)?significant\s+pain\s+points', 'pain_points_summary'),
            (r'(?i)potential\s+product\s+opportunities', 'product_opportunities'),
            (r'(?i)recommendations', 'recommendations'),
            (r'(?i)market\s+trends', 'market_trends'),
            (r'(?i)suggestions\s+for\s+further\s+research', 'further_research')
        ]
        
        # Split text by common section headers
        current_section = 'introduction'
        sections[current_section] = ""
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Check if this line is a section header
            is_header = False
            for pattern, section_name in section_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    current_section = section_name
                    sections[current_section] = ""
                    is_header = True
                    break
            
            # If not a header, add to current section
            if not is_header:
                sections[current_section] += line + "\n"
        
        # Clean up sections (remove extra whitespace)
        for section in sections:
            sections[section] = sections[section].strip()
        
        return sections