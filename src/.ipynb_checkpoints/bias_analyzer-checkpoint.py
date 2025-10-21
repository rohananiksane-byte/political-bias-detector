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
            'cnn.com': {'bias': 6.5, 'credibility': 0.75},
            'msnbc.com': {'bias': 7.5, 'credibility': 0.70},
            'nytimes.com': {'bias': 6.0, 'credibility': 0.85},
            'theguardian.com': {'bias': 6.5, 'credibility': 0.80},
            'foxnews.com': {'bias': 3.5, 'credibility': 0.65},
            'wsj.com': {'bias': 4.5, 'credibility': 0.85},
            'reuters.com': {'bias': 5.0, 'credibility': 0.90},
            'apnews.com': {'bias': 5.0, 'credibility': 0.90},
            'bbc.com': {'bias': 5.0, 'credibility': 0.85},
        }
        
        # Political keywords
        self.left_indicators = [
            'progressive', 'social justice', 'systemic racism', 'climate crisis',
            'wealth inequality', 'workers rights', 'universal healthcare',
            'reproductive rights', 'lgbtq', 'gun control', 'living wage'
        ]
        
        self.right_indicators = [
            'traditional values', 'free market', 'limited government',
            'second amendment', 'border security', 'law and order',
            'pro-life', 'tax cuts', 'deregulation', 'personal responsibility'
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
        
        # Count political indicators
        text_lower = text.lower()
        left_count = sum(1 for word in self.left_indicators if word in text_lower)
        right_count = sum(1 for word in self.right_indicators if word in text_lower)
        
        # Calculate indicator bias
        total = left_count + right_count
        if total > 0:
            indicator_bias = (left_count / total) * 10
        else:
            indicator_bias = 5.0
        
        # Get source bias
        source_info = self.source_ratings.get(source, {'bias': 5.0, 'credibility': 0.70})
        source_bias = source_info['bias']
        credibility = source_info['credibility']
        
        # Calculate final score (weighted average)
        final_bias = (indicator_bias * 0.5) + (source_bias * 0.5)
        
        # Categorize
        if final_bias <= 2.0:
            category = "Hard Right"
        elif final_bias <= 4.0:
            category = "Moderate Right"
        elif final_bias <= 6.0:
            category = "Centre"
        elif final_bias <= 8.0:
            category = "Moderate Left"
        else:
            category = "Hard Left"
        
        # Generate explanation
        explanation = f"This content scored {final_bias:.1f}/10, placing it in the '{category}' category. "
        
        if left_count > right_count:
            explanation += f"Found {left_count} left-leaning indicators vs {right_count} right-leaning. "
        elif right_count > left_count:
            explanation += f"Found {right_count} right-leaning indicators vs {left_count} left-leaning. "
        else:
            explanation += f"Balanced language with {left_count} indicators on each side. "
        
        if source:
            explanation += f"The source '{source}' "
            if source_bias > 6:
                explanation += "is known to lean left. "
            elif source_bias < 4:
                explanation += "is known to lean right. "
            else:
                explanation += "is considered centrist. "
            explanation += f"Source credibility: {credibility:.0%}."
        
        if self.verbose:
            print("✅ Analysis complete!")
        
        return {
            'bias_score': round(final_bias, 2),
            'category': category,
            'credibility_score': round(credibility, 2),
            'explanation': explanation,
            'source': source or 'User text',
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
    print(f"📝 Explanation: {result['explanation']}")