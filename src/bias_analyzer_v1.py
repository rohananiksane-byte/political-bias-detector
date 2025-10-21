"""
Political Bias Analyzer - Core Module
Detects political bias in news articles and text content.
"""

import pandas as pd
import numpy as np
from transformers import pipeline
import warnings
from newspaper import Article
import re
from datetime import datetime

warnings.filterwarnings('ignore')

class BiasAnalyzer:
    """Main class for political bias detection."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print("🚀 Loading Political Bias Analyzer...")
            print("⏳ Loading NLP models (30-60 seconds)...")
        
        # Load sentiment model
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            if self.verbose:
                print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"⚠️  Warning: {e}")
            self.sentiment_analyzer = None
        
        # Load source ratings
        self.source_ratings = {
            'cnn.com': {'bias': 6.5, 'credibility': 0.75, 'name': 'CNN'},
            'msnbc.com': {'bias': 7.5, 'credibility': 0.70, 'name': 'MSNBC'},
            'nytimes.com': {'bias': 6.0, 'credibility': 0.85, 'name': 'New York Times'},
            'theguardian.com': {'bias': 6.5, 'credibility': 0.80, 'name': 'The Guardian'},
            'foxnews.com': {'bias': 3.5, 'credibility': 0.65, 'name': 'Fox News'},
            'wsj.com': {'bias': 4.5, 'credibility': 0.85, 'name': 'Wall Street Journal'},
            'reuters.com': {'bias': 5.0, 'credibility': 0.90, 'name': 'Reuters'},
            'apnews.com': {'bias': 5.0, 'credibility': 0.90, 'name': 'Associated Press'},
            'bbc.com': {'bias': 5.0, 'credibility': 0.85, 'name': 'BBC'},
        }
        
        # EXPANDED Political keywords
        self.left_indicators = [
            'progressive', 'social justice', 'systemic racism', 'systemic inequality',
            'climate crisis', 'climate emergency', 'wealth inequality', 'income inequality',
            'workers rights', 'labor rights', 'union', 'unionize', 'living wage',
            'universal healthcare', 'medicare for all', 'single payer', 'public option',
            'affordable housing', 'rent control', 'minimum wage', 'wage gap',
            'corporate greed', 'corporate accountability', 'tax the rich', 'fair share',
            'reproductive rights', 'abortion rights', 'pro-choice', 'bodily autonomy',
            'lgbtq', 'lgbtqia', 'transgender rights', 'gender identity',
            'marriage equality', 'discrimination', 'marginalized', 'diversity',
            'inclusion', 'equity', 'racial justice', 'police brutality',
            'gun violence', 'gun safety', 'gun control', 'assault weapons ban',
            'renewable energy', 'green energy', 'environmental justice',
            'immigration reform', 'path to citizenship', 'dreamers'
        ]
        
        self.right_indicators = [
            'traditional values', 'free market', 'limited government', 'small government',
            'second amendment', 'border security', 'law and order', 'back the blue',
            'pro-life', 'tax cuts', 'deregulation', 'personal responsibility',
            'fiscal responsibility', 'job creators', 'small business', 'capitalism',
            'religious freedom', 'religious liberty', 'constitutional rights',
            'parental rights', 'states rights', 'individual liberty', 'founding fathers',
            'traditional marriage', 'family values', 'biological sex', 'woke',
            'cancel culture', 'critical race theory', 'illegal immigration',
            'secure the border', 'border wall', 'national security', 'strong military',
            'american values', 'patriotic', 'america first', 'government overreach',
            'government waste', 'balanced budget', 'entitlement reform', 'welfare reform',
            'energy independence', 'oil and gas', 'fracking'
        ]
    
    def extract_from_url(self, url):
        """Extract article from URL."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            return {
                'text': article.text,
                'title': article.title,
                'source': self._extract_domain(url)
            }
        except Exception as e:
            return {'error': f"Failed to extract: {str(e)}"}
    
    def _extract_domain(self, url):
        """Get domain from URL."""
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1) if match else None
    
    def _categorize_bias(self, score):
        """Categorize bias score into political spectrum."""
        if score <= 2.0:
            return "Hard Right"
        elif score <= 4.0:
            return "Moderate Right"
        elif score <= 6.0:
            return "Centre"
        elif score <= 8.0:
            return "Moderate Left"
        else:
            return "Hard Left"
    
    def _generate_explanation(self, final_bias, left_count, right_count, 
                             left_found, right_found, source, source_info):
        """
        Generate detailed 4-5 sentence explanation of the bias analysis.
        """
        category = self._categorize_bias(final_bias)
        source_name = source_info.get('name', source or 'User-provided text')
        source_bias = source_info.get('bias', 5.0)
        credibility = source_info.get('credibility', 0.70)
        
        # Sentence 1: Opening statement with score and category
        explanation = f"This content received a bias score of {final_bias:.1f} out of 10, categorizing it as '{category}'. "
        
        # Sentence 2: Text analysis with ratio
        total_indicators = left_count + right_count
        if total_indicators > 0:
            if left_count > right_count:
                ratio = left_count / right_count if right_count > 0 else left_count
                explanation += f"The textual analysis identified {left_count} left-leaning political indicators compared to {right_count} right-leaning indicators, "
                if ratio >= 3:
                    explanation += f"demonstrating a strong leftward bias in language with a {ratio:.1f}:1 ratio. "
                else:
                    explanation += f"showing a moderate leftward lean in the language used with a {ratio:.1f}:1 ratio. "
            elif right_count > left_count:
                ratio = right_count / left_count if left_count > 0 else right_count
                explanation += f"The textual analysis identified {right_count} right-leaning political indicators compared to {left_count} left-leaning indicators, "
                if ratio >= 3:
                    explanation += f"demonstrating a strong rightward bias in language with a {ratio:.1f}:1 ratio. "
                else:
                    explanation += f"showing a moderate rightward lean in the language used with a {ratio:.1f}:1 ratio. "
            else:
                explanation += f"The text contains an equal number of left and right political indicators ({left_count} each), suggesting balanced or centrist language choices. "
        else:
            explanation += "No strong political indicators from either side of the spectrum were detected in the textual analysis, suggesting neutral or non-political language. "
        
        # Sentence 3: Specific examples of keywords found
        if left_found or right_found:
            explanation += "Specific terminology includes: "
            if left_found:
                top_left = left_found[:3]
                explanation += f"left-leaning phrases such as '{top_left[0]}'"
                if len(top_left) > 1:
                    explanation += f", '{top_left[1]}'"
                if len(top_left) > 2:
                    explanation += f", and '{top_left[2]}'"
                if right_found:
                    explanation += "; alongside "
            if right_found:
                top_right = right_found[:3]
                explanation += f"right-leaning phrases such as '{top_right[0]}'"
                if len(top_right) > 1:
                    explanation += f", '{top_right[1]}'"
                if len(top_right) > 2:
                    explanation += f", and '{top_right[2]}'"
            explanation += ". "
        
        # Sentence 4: Source reputation analysis
        if source and source_name not in ['User-provided text', 'User text', 'Unknown']:
            explanation += f"The source '{source_name}' "
            if source_bias > 6.5:
                explanation += "is recognized in media bias research as having a left-leaning editorial perspective, which influenced the overall assessment. "
            elif source_bias >= 5.5:
                explanation += "is recognized as having a slight left-of-center editorial perspective, though it maintains relatively balanced reporting standards. "
            elif source_bias >= 4.5:
                explanation += "is considered a relatively centrist news organization with balanced editorial standards across the political spectrum. "
            elif source_bias >= 3.5:
                explanation += "is recognized as having a slight right-of-center editorial perspective, though it maintains professional journalistic standards. "
            else:
                explanation += "is recognized in media bias research as having a right-leaning editorial perspective, which influenced the overall assessment. "
            
            explanation += f"This outlet has a credibility rating of {credibility:.0%} based on factual accuracy, editorial transparency, and adherence to journalistic standards. "
        
        # Sentence 5: Final interpretation
        if final_bias >= 7.5:
            explanation += "Overall, this content exhibits characteristics typical of progressive or liberal political commentary, emphasizing themes of social equity, government intervention, and systemic reform."
        elif final_bias >= 6.0:
            explanation += "Overall, this content shows a moderate liberal orientation, incorporating progressive perspectives while maintaining some analytical balance."
        elif final_bias >= 4.5:
            explanation += "Overall, this content demonstrates a centrist approach, presenting viewpoints from across the political spectrum without strong ideological positioning."
        elif final_bias >= 2.5:
            explanation += "Overall, this content shows a moderate conservative orientation, emphasizing traditional approaches and market-based solutions while maintaining analytical nuance."
        else:
            explanation += "Overall, this content exhibits characteristics typical of conservative or right-wing political commentary, emphasizing individual liberty, limited government, and traditional institutional frameworks."
        
        return explanation
    
    def analyze(self, input_content):
        """Analyze text or URL for political bias."""
        
        # Check if URL
        is_url = input_content.startswith('http')
        
        if is_url:
            if self.verbose:
                print("📰 Extracting article...")
            article_data = self.extract_from_url(input_content)
            
            if 'error' in article_data:
                return article_data
            
            text = article_data['text']
            source = article_data.get('source')
            title = article_data.get('title')
        else:
            text = input_content
            source = None
            title = None
        
        if self.verbose:
            print(f"📊 Analyzing {len(text)} characters...")
        
        # Count political indicators and track which ones were found
        text_lower = text.lower()
        left_count = sum(1 for word in self.left_indicators if word in text_lower)
        right_count = sum(1 for word in self.right_indicators if word in text_lower)
        
        # Track specific phrases found
        left_found = [phrase for phrase in self.left_indicators if phrase in text_lower]
        right_found = [phrase for phrase in self.right_indicators if phrase in text_lower]
        
        # Calculate indicator bias
        total = left_count + right_count
        if total > 0:
            indicator_bias = (left_count / total) * 10
        else:
            indicator_bias = 5.0
        
        # Get source bias
        source_info = self.source_ratings.get(source, {
            'bias': 5.0, 
            'credibility': 0.70,
            'name': source or 'User text'
        })
        source_bias = source_info['bias']
        credibility = source_info['credibility']
        
        # Calculate final score (weighted average)
        final_bias = (indicator_bias * 0.5) + (source_bias * 0.5)
        
        # Get category
        category = self._categorize_bias(final_bias)
        
        # Generate detailed explanation
        explanation = self._generate_explanation(
            final_bias, left_count, right_count,
            left_found, right_found, source, source_info
        )
        
        if self.verbose:
            print("✅ Analysis complete!")
        
        return {
            'bias_score': round(final_bias, 2),
            'category': category,
            'credibility_score': round(credibility, 2),
            'explanation': explanation,
            'source': source_info.get('name', source or 'User text'),
            'title': title,
            'left_indicators': left_count,
            'right_indicators': right_count
        }


# Test code
if __name__ == "__main__":
    print("="*60)
    print("TESTING BIAS ANALYZER")
    print("="*60)
    
    analyzer = BiasAnalyzer()
    
    # Test text
    test_text = """
    The new policy promotes social justice and addresses systemic 
    inequality. Progressive leaders support universal healthcare 
    and workers rights for all citizens.
    """
    
    result = analyzer.analyze(test_text)
    
    print(f"\n📊 Score: {result['bias_score']}/10")
    print(f"📍 Category: {result['category']}")
    print(f"📝 Explanation:\n{result['explanation']}")