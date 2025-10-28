"""
Enhanced Contextual Bias Analyzer with AI Sentiment Detection
Uses transformer models to understand context around political keywords
"""

import pandas as pd
import numpy as np
from transformers import pipeline
import warnings
from newspaper import Article
import re
from datetime import datetime

warnings.filterwarnings('ignore')

class EnhancedContextualAnalyzer:
    """Analyzer with AI-powered sentiment detection for context."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print("🚀 Loading Enhanced Contextual Bias Analyzer with AI Sentiment...")
            print("⏳ Loading NLP models (this may take 60-90 seconds)...")
        
        # Load sentiment model for context analysis
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            if self.verbose:
                print("✅ Sentiment model loaded!")
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
        
        # Political keywords (concepts, not stances)
        self.left_concepts = {
            'progressive', 'social justice', 'systemic racism', 'systemic inequality',
            'climate crisis', 'climate emergency', 'wealth inequality', 'income inequality',
            'workers rights', 'labor rights', 'union', 'unionize', 'living wage',
            'universal healthcare', 'medicare for all', 'single payer', 'public option',
            'affordable housing', 'rent control', 'minimum wage', 'wage gap',
            'corporate greed', 'corporate accountability', 'tax the rich', 'fair share',
            'reproductive rights', 'abortion rights', 'pro-choice', 'bodily autonomy',
            'lgbtq', 'lgbtqia', 'transgender rights', 'gender identity', 'gay rights',
            'marriage equality', 'discrimination', 'marginalized', 'diversity',
            'inclusion', 'equity', 'racial justice', 'police brutality',
            'gun violence', 'gun safety', 'gun control', 'assault weapons ban',
            'renewable energy', 'green energy', 'environmental justice',
            'immigration reform', 'path to citizenship', 'dreamers', 'asylum'
        }
        
        self.right_concepts = {
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
            'energy independence', 'oil and gas', 'fracking', 'drill'
        }
        
        # MASSIVELY EXPANDED negative indicator words
        self.negative_indicators = {
            # Direct opposition
            'ban', 'banned', 'banning', 'prohibit', 'prohibited', 'against', 'oppose',
            'opposed', 'opposing', 'opposition', 'stop', 'stopped', 'end', 'ended',
            'eliminate', 'eliminated', 'remove', 'removed', 'reject', 'rejected',
            'deny', 'denied', 'denying', 'restrict', 'restricted', 'block', 'blocked',
            'prevent', 'prevented', 'preventing', 'illegal', 'unlawful', 'forbidden',
            
            # Negative judgment
            'wrong', 'bad', 'harmful', 'dangerous', 'threat', 'threatens', 'threatening',
            'evil', 'terrible', 'horrible', 'awful', 'disgusting', 'unacceptable',
            'immoral', 'unethical', 'corrupt', 'corrupted', 'criminal', 'crime',
            
            # Negation
            'should not', 'must not', 'cannot', 'can\'t', 'won\'t', 'never', 'no',
            'not', 'anti', 'destroy', 'destroyed', 'destroying', 'destruction',
            'abolish', 'abolished', 'repeal', 'repealed', 'reverse', 'reversed',
            'undo', 'fight against', 'fighting against',
            
            # Harm/punishment
            'jail', 'jailed', 'imprison', 'imprisoned', 'arrest', 'arrested',
            'punish', 'punished', 'punishment', 'penalize', 'fine', 'fined',
            'detain', 'detained', 'deport', 'deported', 'execute', 'executed',
            'kill', 'killing', 'death', 'murder',
            
            # Exclusion
            'exclude', 'excluded', 'excluding', 'ban', 'expel', 'expelled',
            'kick out', 'get rid of', 'eliminate', 'segregate', 'segregated',
            'separate', 'separated', 'discriminate', 'discrimination',
            
            # Restriction
            'limit', 'limited', 'limiting', 'curtail', 'curtailed', 'suppress',
            'suppressed', 'censor', 'censored', 'silence', 'silenced',
            
            # Negative emotion
            'hate', 'hated', 'despise', 'despised', 'loathe', 'disgusted',
            'fear', 'scared', 'afraid', 'worried', 'concerned',
            
            # Dismissal
            'ridiculous', 'absurd', 'nonsense', 'stupid', 'idiotic', 'foolish',
            'crazy', 'insane', 'delusional'
        }
        
        # Positive indicator words
        self.positive_indicators = {
            'support', 'supports', 'supported', 'supporting', 'promote', 'promotes',
            'promoted', 'promoting', 'advocate', 'advocates', 'advocating', 'defend',
            'defends', 'defending', 'protect', 'protects', 'protected', 'protecting',
            'champion', 'champions', 'embrace', 'embraces', 'embracing', 'celebrate',
            'celebrates', 'celebrating', 'expand', 'expands', 'expanding', 'strengthen',
            'strengthens', 'ensure', 'ensures', 'guarantee', 'guarantees', 'enable',
            'enables', 'empower', 'empowers', 'empowering', 'encourage', 'encourages',
            'should', 'must', 'need to', 'have to', 'right to', 'important', 'essential',
            'necessary', 'crucial', 'vital', 'good', 'great', 'excellent', 'beneficial',
            'positive', 'wonderful', 'amazing', 'love', 'respect', 'honor', 'value',
            'cherish', 'appreciate', 'welcome', 'accept', 'include', 'embrace'
        }
    
    def _extract_sentences_with_keyword(self, text, keyword):
        """Extract full sentences containing the keyword."""
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                relevant_sentences.append(sentence.strip())
        
        return relevant_sentences
    
    def _analyze_sentence_sentiment_ai(self, sentence):
        """Use AI model to determine if sentence is positive or negative."""
        if not self.sentiment_analyzer or not sentence:
            return 0.0
        
        try:
            result = self.sentiment_analyzer(sentence[:512])[0]  # Truncate to model limit
            
            # Convert to score: negative = -1, positive = +1
            if result['label'] == 'NEGATIVE':
                return -result['score']
            else:
                return result['score']
        except:
            return 0.0
    
    def _analyze_keyword_context(self, text, keyword, is_left_concept):
        """
        Analyze context around a keyword using multiple methods.
        Returns: 'support', 'oppose', or 'neutral'
        """
        # Method 1: Get sentences containing the keyword
        sentences = self._extract_sentences_with_keyword(text, keyword)
        
        if not sentences:
            return 'neutral', 0.5
        
        # Method 2: Check for explicit positive/negative words
        text_lower = ' '.join(sentences).lower()
        
        negative_count = sum(1 for word in self.negative_indicators if word in text_lower)
        positive_count = sum(1 for word in self.positive_indicators if word in text_lower)
        
        # Method 3: Use AI sentiment analysis
        ai_sentiments = [self._analyze_sentence_sentiment_ai(s) for s in sentences]
        avg_ai_sentiment = np.mean(ai_sentiments) if ai_sentiments else 0.0
        
        # Combine methods (weighted)
        word_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        combined_score = (word_score * 0.4) + (avg_ai_sentiment * 0.6)
        
        # Determine stance
        confidence = abs(combined_score)
        
        if combined_score > 0.2:
            return 'support', confidence
        elif combined_score < -0.2:
            return 'oppose', confidence
        else:
            return 'neutral', confidence
    
    def _contextual_analysis(self, text):
        """Perform full contextual analysis."""
        left_support = 0
        left_oppose = 0
        right_support = 0
        right_oppose = 0
        
        context_notes = []
        left_examples = []
        right_examples = []
        
        text_lower = text.lower()
        
        # Analyze LEFT concepts
        for keyword in self.left_concepts:
            if keyword in text_lower:
                stance, confidence = self._analyze_keyword_context(text, keyword, True)
                
                if stance == 'support':
                    left_support += confidence
                    left_examples.append(f"support for {keyword}")
                    context_notes.append(f"✓ Support for '{keyword}' (left-leaning)")
                elif stance == 'oppose':
                    right_support += confidence  # Opposition to left = right
                    right_examples.append(f"opposition to {keyword}")
                    context_notes.append(f"✗ Opposition to '{keyword}' (right-leaning)")
                else:
                    left_support += confidence * 0.3  # Weak neutral mention
        
        # Analyze RIGHT concepts
        for keyword in self.right_concepts:
            if keyword in text_lower:
                stance, confidence = self._analyze_keyword_context(text, keyword, False)
                
                if stance == 'support':
                    right_support += confidence
                    right_examples.append(f"support for {keyword}")
                    context_notes.append(f"✓ Support for '{keyword}' (right-leaning)")
                elif stance == 'oppose':
                    left_support += confidence  # Opposition to right = left
                    left_examples.append(f"opposition to {keyword}")
                    context_notes.append(f"✗ Opposition to '{keyword}' (left-leaning)")
                else:
                    right_support += confidence * 0.3
        
        return {
            'left_score': left_support,
            'right_score': right_support,
            'left_examples': left_examples[:5],
            'right_examples': right_examples[:5],
            'context_notes': context_notes[:5]
        }
    
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
        """Categorize bias score."""
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
    
    def _generate_explanation(self, final_bias, analysis, source, source_info):
        """Generate detailed explanation."""
        category = self._categorize_bias(final_bias)
        source_name = source_info.get('name', source or 'User-provided text')
        
        left_score = analysis['left_score']
        right_score = analysis['right_score']
        context_notes = analysis['context_notes']
        
        explanation = f"This content received a bias score of {final_bias:.1f} out of 10, categorizing it as '{category}'. "
        
        total = left_score + right_score
        if total > 0:
            if left_score > right_score:
                ratio = left_score / right_score if right_score > 0 else left_score
                explanation += f"The AI-powered contextual analysis identified {left_score:.1f} points of left-leaning positioning compared to {right_score:.1f} points of right-leaning positioning (ratio: {ratio:.1f}:1). "
            elif right_score > left_score:
                ratio = right_score / left_score if left_score > 0 else right_score
                explanation += f"The AI-powered contextual analysis identified {right_score:.1f} points of right-leaning positioning compared to {left_score:.1f} points of left-leaning positioning (ratio: {ratio:.1f}:1). "
            else:
                explanation += "The analysis found balanced political positioning. "
        
        if context_notes:
            explanation += "Key contextual findings: " + "; ".join(context_notes[:3]) + ". "
        
        if source and source_name not in ['User-provided text', 'User text']:
            source_bias = source_info.get('bias', 5.0)
            credibility = source_info.get('credibility', 0.70)
            
            if source_bias > 6.5:
                explanation += f"The source '{source_name}' has a left-leaning editorial perspective. "
            elif source_bias < 3.5:
                explanation += f"The source '{source_name}' has a right-leaning editorial perspective. "
            else:
                explanation += f"The source '{source_name}' is considered relatively centrist. "
            
            explanation += f"Credibility: {credibility:.0%}. "
        
        if final_bias >= 7.5:
            explanation += "Overall, this exhibits progressive political framing emphasizing social equity and reform."
        elif final_bias >= 6.0:
            explanation += "Overall, this shows moderate liberal positioning with some balance."
        elif final_bias >= 4.5:
            explanation += "Overall, this demonstrates a centrist approach."
        elif final_bias >= 2.5:
            explanation += "Overall, this shows moderate conservative positioning."
        else:
            explanation += "Overall, this exhibits conservative framing emphasizing individual liberty and limited government."
        
        return explanation
    
    def analyze(self, input_content):
        """Main analysis function with enhanced context detection."""
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
            print(f"📊 Analyzing with AI-powered context detection...")
        
        # Perform enhanced contextual analysis
        analysis = self._contextual_analysis(text)
        
        left_score = analysis['left_score']
        right_score = analysis['right_score']
        
        # Calculate indicator bias
        total = left_score + right_score
        if total > 0:
            indicator_bias = (left_score / total) * 10
        else:
            indicator_bias = 5.0
        
        # Get source bias
        source_info = self.source_ratings.get(source, {
            'bias': 5.0, 'credibility': 0.70, 'name': source or 'User text'
        })
        source_bias = source_info['bias']
        credibility = source_info['credibility']
        
        # Calculate final score
        final_bias = (indicator_bias * 0.6) + (source_bias * 0.4)
        
        category = self._categorize_bias(final_bias)
        explanation = self._generate_explanation(final_bias, analysis, source, source_info)
        
        if self.verbose:
            print("✅ AI contextual analysis complete!")
        
        return {
            'bias_score': round(final_bias, 2),
            'category': category,
            'credibility_score': round(credibility, 2),
            'explanation': explanation,
            'source': source_info.get('name', source or 'User text'),
            'title': title,
            'left_indicators': round(left_score, 1),
            'right_indicators': round(right_score, 1),
            'context_notes': analysis['context_notes']
        }


# Test cases
if __name__ == "__main__":
    print("="*70)
    print("TESTING ENHANCED AI CONTEXT ANALYZER")
    print("="*70)
    
    analyzer = EnhancedContextualAnalyzer()
    
    # Test 1: LGBTQ + jail (your example)
    print("\n" + "="*70)
    print("TEST 1: 'LGBTQ people should be jailed'")
    print("="*70)
    test1 = "LGBTQ people should be jailed"
    result1 = analyzer.analyze(test1)
    print(f"\n📊 Score: {result1['bias_score']}/10")
    print(f"📍 Category: {result1['category']}")
    print(f"Context: {result1['context_notes']}")
    
    # Test 2: LGBTQ + support
    print("\n" + "="*70)
    print("TEST 2: 'We must support and protect LGBTQ rights'")
    print("="*70)
    test2 = "We must support and protect LGBTQ rights for everyone"
    result2 = analyzer.analyze(test2)
    print(f"\n📊 Score: {result2['bias_score']}/10")
    print(f"📍 Category: {result2['category']}")
    print(f"Context: {result2['context_notes']}")
    
    # Test 3: Gun control + ban
    print("\n" + "="*70)
    print("TEST 3: 'Gun control should be banned'")
    print("="*70)
    test3 = "Gun control should be banned and eliminated"
    result3 = analyzer.analyze(test3)
    print(f"\n📊 Score: {result3['bias_score']}/10")
    print(f"📍 Category: {result3['category']}")
    print(f"Context: {result3['context_notes']}")
    
    print("\n" + "="*70)
    print("✅ Enhanced AI context analysis complete!")
    print("="*70)