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
            print(f"⚠️ Warning: {e}")
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
            'climate crisis', 'climate emergency', 'climate change', 'global warming',
            'climate action', 'carbon emissions', 'greenhouse gas',
            'wealth inequality', 'income inequality', 'economic inequality',
            'workers rights', 'labor rights', 'union', 'unionize', 'living wage',
            'universal healthcare', 'medicare for all', 'single payer', 'public option',
            'affordable housing', 'rent control', 'minimum wage', 'wage gap', 'living wage',
            'corporate greed', 'corporate accountability', 'tax the rich', 'fair share',
            'wealth tax', 'corporate taxes', 'tax justice',
            'reproductive rights', 'abortion rights', 'pro-choice', 'bodily autonomy',
            'womens rights', 'gender equality', 'pay equity',
            'lgbtq', 'lgbtqia', 'transgender rights', 'gender identity', 'gay rights',
            'marriage equality', 'discrimination', 'marginalized', 'diversity',
            'inclusion', 'equity', 'racial justice', 'police brutality', 'police reform',
            'criminal justice reform', 'mass incarceration', 'prison reform',
            'gun violence', 'gun safety', 'gun control', 'assault weapons ban',
            'background checks', 'gun reform',
            'renewable energy', 'green energy', 'environmental justice', 'solar power',
            'wind power', 'clean energy', 'sustainable',
            'immigration reform', 'path to citizenship', 'dreamers', 'asylum',
            'refugee', 'undocumented', 'immigrant rights',
            'voting rights', 'voter suppression', 'gerrymandering', 'electoral reform',
            'healthcare access', 'public education', 'student debt', 'tuition free',
            'social safety net', 'welfare', 'food stamps', 'medicaid expansion',
            'paid family leave', 'childcare', 'universal basic income',
            'anti-discrimination', 'hate crime', 'white supremacy', 'institutional racism'
        }
        
        self.right_concepts = {
            'traditional values', 'free market', 'limited government', 'small government',
            'second amendment', '2nd amendment', 'gun rights', 'right to bear arms',
            'border security', 'law and order', 'back the blue', 'support police',
            'pro-life', 'unborn', 'right to life', 'sanctity of life',
            'tax cuts', 'tax relief', 'lower taxes', 'deregulation', 'personal responsibility',
            'fiscal responsibility', 'fiscal conservative', 'job creators', 'small business', 
            'capitalism', 'free enterprise', 'market economy',
            'religious freedom', 'religious liberty', 'constitutional rights',
            'parental rights', 'school choice', 'states rights', 'individual liberty', 
            'founding fathers', 'constitution',
            'traditional marriage', 'family values', 'biological sex', 'woke',
            'cancel culture', 'critical race theory', 'crt', 'indoctrination',
            'illegal immigration', 'illegal aliens', 'border crisis',
            'secure the border', 'border wall', 'national security', 'strong military',
            'military strength', 'defense spending',
            'american values', 'patriotic', 'patriotism', 'america first', 
            'government overreach', 'government waste', 'bureaucracy',
            'balanced budget', 'national debt', 'deficit reduction',
            'entitlement reform', 'welfare reform', 'food stamp reform',
            'energy independence', 'oil and gas', 'fracking', 'drill',
            'fossil fuels', 'coal', 'natural gas',
            'law and order', 'tough on crime', 'crime prevention',
            'voter id', 'election integrity', 'election security',
            'school prayer', 'religious values', 'judeo-christian',
            'pro-business', 'economic growth', 'job growth',
            'self-reliance', 'rugged individualism', 'bootstraps',
            'sovereign', 'sovereignty', 'nationalism'
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
            'should not', 'must not', 'cannot', "can't", "won't", 'never', 'no',
            'not', 'anti', 'destroy', 'destroyed', 'destroying', 'destruction',
            'abolish', 'abolished', 'repeal', 'repealed', 'reverse', 'reversed',
            'undo', 'fight against', 'fighting against',
            
            # Harm/punishment
            'jail', 'jailed', 'imprison', 'imprisoned', 'arrest', 'arrested',
            'punish', 'punished', 'punishment', 'penalize', 'fine', 'fined',
            'detain', 'detained', 'deport', 'deported', 'execute', 'executed',
            'kill', 'killing', 'death', 'murder',
            
            # Exclusion
            'exclude', 'excluded', 'excluding', 'expel', 'expelled',
            'kick out', 'get rid of', 'segregate', 'segregated',
            'separate', 'separated', 'discriminate',
            
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
            'cherish', 'appreciate', 'welcome', 'accept', 'include'
        }
    
    def _extract_sentences_with_keyword(self, text, keyword):
        """Extract full sentences containing the keyword."""
        # Better sentence splitting that handles more cases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip empty or very short sentences
            if len(sentence) < 10:
                continue
            if keyword.lower() in sentence.lower():
                # Clean up the sentence
                sentence = re.sub(r'\s+', ' ', sentence)  # Normalize whitespace
                relevant_sentences.append(sentence)
                if self.verbose and len(relevant_sentences) == 1:
                    print(f"      📝 Example: {sentence[:100]}...")
        
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
        Returns: 'support', 'oppose', or 'neutral', confidence, and example sentences
        """
        # Method 1: Get sentences containing the keyword
        sentences = self._extract_sentences_with_keyword(text, keyword)
        
        if not sentences:
            if self.verbose:
                print(f"      ⚠️ Keyword '{keyword}' found but no sentences extracted")
            return 'neutral', 0.5, []
        
        if self.verbose:
            print(f"      ✓ Extracted {len(sentences)} sentence(s) for '{keyword}'")
        
        # Method 2: Check for explicit positive/negative words in surrounding context
        text_lower = ' '.join(sentences).lower()
        
        # Count indicators with position weighting (closer to keyword = more weight)
        negative_matches = []
        positive_matches = []
        
        for word in self.negative_indicators:
            if word in text_lower:
                negative_matches.append(word)
        
        for word in self.positive_indicators:
            if word in text_lower:
                positive_matches.append(word)
        
        negative_count = len(negative_matches)
        positive_count = len(positive_matches)
        
        # Method 3: Use AI sentiment analysis on each sentence
        ai_sentiments = []
        for s in sentences:
            sentiment = self._analyze_sentence_sentiment_ai(s)
            ai_sentiments.append(sentiment)
        
        avg_ai_sentiment = np.mean(ai_sentiments) if ai_sentiments else 0.0
        
        # Method 4: Check for negation phrases near the keyword
        negation_phrases = ['not', 'never', 'no', "don't", "doesn't", 'without', 'against', 'oppose', 'reject']
        has_nearby_negation = any(neg in text_lower for neg in negation_phrases)
        
        # Combine methods with weighted scoring
        # Word indicators: 30%, AI sentiment: 50%, Negation context: 20%
        word_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        
        # Apply negation modifier
        negation_modifier = -0.3 if has_nearby_negation and positive_count > negative_count else 0
        
        combined_score = (word_score * 0.3) + (avg_ai_sentiment * 0.5) + negation_modifier
        
        # Determine stance with adjusted thresholds
        confidence = min(abs(combined_score), 1.0)  # Cap at 1.0
        
        if self.verbose:
            print(f"      📊 Word score: {word_score:.2f}, AI sentiment: {avg_ai_sentiment:.2f}")
            print(f"      📊 Combined: {combined_score:.2f}, Confidence: {confidence:.2f}")
            if negative_matches:
                print(f"      🔍 Negative indicators: {', '.join(negative_matches[:3])}")
            if positive_matches:
                print(f"      🔍 Positive indicators: {', '.join(positive_matches[:3])}")
        
        # More nuanced threshold for stance determination
        if combined_score > 0.15:  # Lower threshold for support
            return 'support', confidence, sentences[:3]
        elif combined_score < -0.15:  # Lower threshold for oppose
            return 'oppose', confidence, sentences[:3]
        else:
            return 'neutral', confidence * 0.5, sentences[:2]  # Neutral gets lower confidence
    
    def _contextual_analysis(self, text):
        """Perform full contextual analysis."""
        left_support = 0
        left_oppose = 0
        right_support = 0
        right_oppose = 0
        
        context_notes = []
        left_examples = []
        right_examples = []
        detailed_findings = []  # NEW: Store detailed findings with examples
        
        text_lower = text.lower()
        
        # Analyze LEFT concepts
        for keyword in self.left_concepts:
            if keyword in text_lower:
                stance, confidence, example_sentences = self._analyze_keyword_context(text, keyword, True)
                
                if stance == 'support':
                    left_support += confidence
                    left_examples.append(f"support for {keyword}")
                    context_notes.append(f"✓ Support for '{keyword}' (left-leaning)")
                    # Add detailed finding
                    if example_sentences:
                        detailed_findings.append({
                            'term': keyword,
                            'stance': 'support',
                            'direction': 'left',
                            'example': example_sentences[0][:150] + '...' if len(example_sentences[0]) > 150 else example_sentences[0]
                        })
                elif stance == 'oppose':
                    right_support += confidence  # Opposition to left = right
                    right_examples.append(f"opposition to {keyword}")
                    context_notes.append(f"✗ Opposition to '{keyword}' (right-leaning)")
                    if example_sentences:
                        detailed_findings.append({
                            'term': keyword,
                            'stance': 'oppose',
                            'direction': 'right',
                            'example': example_sentences[0][:150] + '...' if len(example_sentences[0]) > 150 else example_sentences[0]
                        })
                else:
                    left_support += confidence * 0.3  # Weak neutral mention
        
        # Analyze RIGHT concepts
        for keyword in self.right_concepts:
            if keyword in text_lower:
                stance, confidence, example_sentences = self._analyze_keyword_context(text, keyword, False)
                
                if stance == 'support':
                    right_support += confidence
                    right_examples.append(f"support for {keyword}")
                    context_notes.append(f"✓ Support for '{keyword}' (right-leaning)")
                    if example_sentences:
                        detailed_findings.append({
                            'term': keyword,
                            'stance': 'support',
                            'direction': 'right',
                            'example': example_sentences[0][:150] + '...' if len(example_sentences[0]) > 150 else example_sentences[0]
                        })
                elif stance == 'oppose':
                    left_support += confidence  # Opposition to right = left
                    left_examples.append(f"opposition to {keyword}")
                    context_notes.append(f"✗ Opposition to '{keyword}' (left-leaning)")
                    if example_sentences:
                        detailed_findings.append({
                            'term': keyword,
                            'stance': 'oppose',
                            'direction': 'left',
                            'example': example_sentences[0][:150] + '...' if len(example_sentences[0]) > 150 else example_sentences[0]
                        })
                else:
                    right_support += confidence * 0.3
        
        return {
            'left_score': left_support,
            'right_score': right_support,
            'left_examples': left_examples[:5],
            'right_examples': right_examples[:5],
            'context_notes': context_notes[:10],
            'detailed_findings': detailed_findings[:8]  # NEW: Return detailed findings
        }
    
    def extract_from_url(self, url):
        """Extract article from URL."""
        try:
            if self.verbose:
                print(f"🌐 Downloading article from: {url}")
            
            article = Article(url)
            article.download()
            article.parse()
            
            text = article.text
            if self.verbose:
                print(f"✅ Article extracted: {len(text)} characters")
                print(f"📰 Title: {article.title}")
                print(f"🔤 First 200 chars: {text[:200]}...")
            
            return {
                'text': text,
                'title': article.title,
                'source': self._extract_domain(url)
            }
        except Exception as e:
            if self.verbose:
                print(f"❌ Extraction error: {str(e)}")
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
    
    def _generate_explanation(self, final_bias, analysis, source, source_info, indicator_bias=5.0):
        """Generate detailed explanation with specific examples."""
        category = self._categorize_bias(final_bias)
        source_name = source_info.get('name', source or 'User-provided text')
        
        left_score = analysis['left_score']
        right_score = analysis['right_score']
        detailed_findings = analysis.get('detailed_findings', [])
        
        explanation = f"This content received a bias score of {final_bias:.1f} out of 10, categorizing it as '{category}'. "
        
        # Add specific findings with examples
        total = left_score + right_score
        if total > 0:
            if left_score > right_score:
                ratio = left_score / right_score if right_score > 0 else left_score
                explanation += f"The AI detected {left_score:.1f} points of left-leaning positioning compared to {right_score:.1f} points of right-leaning positioning (ratio: {ratio:.1f}:1). "
            elif right_score > left_score:
                ratio = right_score / left_score if left_score > 0 else right_score
                explanation += f"The AI detected {right_score:.1f} points of right-leaning positioning compared to {left_score:.1f} points of left-leaning positioning (ratio: {ratio:.1f}:1). "
            else:
                explanation += "The analysis found relatively balanced political positioning. "
        
        # NEW: Add specific term examples - ALWAYS show if we have them
        if detailed_findings and len(detailed_findings) > 0:
            explanation += "\n\n📌 Specific Political Terms Identified:\n"
            for i, finding in enumerate(detailed_findings[:6], 1):
                stance_verb = "SUPPORTS" if finding['stance'] == 'support' else "OPPOSES"
                direction_emoji = "🔵" if finding['direction'] == 'left' else "🔴"
                explanation += f"\n{direction_emoji} {i}. {stance_verb} '{finding['term']}'\n   Quote: \"{finding['example']}\"\n"
        elif total > 0:
            # Fallback if no detailed findings but we detected terms
            explanation += "\n\n⚠️ Political language detected but specific examples could not be extracted."
        
        # Source information
        if source and source_name not in ['User-provided text', 'User text']:
            source_bias = source_info.get('bias', 5.0)
            credibility = source_info.get('credibility', 0.70)
            
            explanation += f"\n\nSource Analysis: "
            if source_bias > 6.5:
                explanation += f"'{source_name}' has a known left-leaning editorial perspective. "
            elif source_bias < 3.5:
                explanation += f"'{source_name}' has a known right-leaning editorial perspective. "
            else:
                explanation += f"'{source_name}' is considered relatively centrist. "
            
            explanation += f"Credibility rating: {credibility:.0%}. "
        
        # Overall assessment
        explanation += "\n\nOverall Assessment: "
        if final_bias >= 7.5:
            explanation += "This content exhibits strong progressive framing, emphasizing social equity, government intervention, and systemic reform."
        elif final_bias >= 6.0:
            explanation += "This content shows moderate liberal positioning with emphasis on social programs and regulatory oversight."
        elif final_bias >= 4.5:
            explanation += "This content demonstrates a centrist or balanced approach to political issues."
        elif final_bias >= 2.5:
            explanation += "This content shows moderate conservative positioning with emphasis on traditional values and limited government."
        else:
            explanation += "This content exhibits strong conservative framing, emphasizing individual liberty, free markets, and traditional institutions."
        
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
            print(f"📄 Text length: {len(text)} characters")
        
        # Perform enhanced contextual analysis
        analysis = self._contextual_analysis(text)
        
        if self.verbose:
            print(f"🔍 Found {len(analysis.get('detailed_findings', []))} detailed term matches")
        
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
        explanation = self._generate_explanation(final_bias, analysis, source, source_info, indicator_bias)
        
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
            'context_notes': analysis['context_notes'],
            'detailed_findings': analysis.get('detailed_findings', [])  # NEW
        }


# Test cases
if __name__ == "__main__":
    print("="*70)
    print("TESTING ENHANCED AI CONTEXT ANALYZER")
    print("="*70)
    
    analyzer = EnhancedContextualAnalyzer()
    
    # Test 1: Climate change support
    print("\n" + "="*70)
    print("TEST 1: Climate change article")
    print("="*70)
    test1 = """Climate change is an urgent crisis that requires immediate action. 
    We must invest in renewable energy and implement strong environmental regulations 
    to protect our planet for future generations. The wealthy must pay their fair share 
    to fund green energy initiatives."""
    result1 = analyzer.analyze(test1)
    print(f"\n📊 Score: {result1['bias_score']}/10")
    print(f"📁 Category: {result1['category']}")
    print(f"📝 Explanation:\n{result1['explanation']}")
    
    # Test 2: Conservative viewpoint
    print("\n" + "="*70)
    print("TEST 2: Conservative article")
    print("="*70)
    test2 = """Traditional values and limited government are essential for protecting 
    individual liberty. We must secure our borders, support our police, and reduce 
    government overreach. Free markets and personal responsibility are the foundation 
    of prosperity."""
    result2 = analyzer.analyze(test2)
    print(f"\n📊 Score: {result2['bias_score']}/10")
    print(f"📁 Category: {result2['category']}")
    print(f"📝 Explanation:\n{result2['explanation']}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)