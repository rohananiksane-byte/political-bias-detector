"""
Flask API for Bias Detection
Provides REST API endpoints for the bias analyzer.
"""
import os
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from bias_analyzer import EnhancedContextualAnalyzer as BiasAnalyzer
import logging

app = Flask(__name__)

# Enable CORS for all origins (allows any website to call your API)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Allow all origins for public API
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize analyzer
analyzer = None

def get_analyzer():
    """Lazy load analyzer to avoid initialization on module import."""
    global analyzer
    if analyzer is None:
        logger.info("Initializing analyzer...")
        try:
            analyzer = BiasAnalyzer(verbose=False)
            logger.info("Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {str(e)}")
            raise
    return analyzer

@app.route('/')
def home():
    """API info page."""
    return jsonify({
        'name': 'Political Bias Detection API',
        'version': '1.0',
        'status': 'online',
        'endpoints': {
            '/api/analyze': 'POST - Analyze text or URL',
            '/api/health': 'GET - Check API status'
        },
        'documentation': 'https://github.com/rohananiksane-byte/political-bias-detector'
    })

@app.route('/api/health')
@app.route('/health')  # Add alternative route without /api prefix
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': str(datetime.datetime.now()),
        'analyzer_loaded': analyzer is not None
    })

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])  # Added OPTIONS for CORS preflight
def analyze():
    """
    Analyze content for political bias.
    
    Request body:
    {
        "content": "text or URL to analyze"
    }
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing content field'
            }), 400
        
        content = data['content']
        
        if not content or len(content.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Content cannot be empty'
            }), 400
        
        # Get analyzer and run analysis
        logger.info(f"Analyzing content (length: {len(content)} chars)")
        analyzer_instance = get_analyzer()
        result = analyzer_instance.analyze(content)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        logger.info(f"Analysis complete - Score: {result.get('bias_score')}")
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An error occurred during analysis. Please try again.'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/api/analyze', '/api/health', '/health', '/']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Get port from environment variable (for Render) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("🚀 STARTING BIAS DETECTION API")
    print("="*60)
    print(f"📍 API running on port: {port}")
    
    # Check if running locally or in production
    if os.environ.get('PORT'):
        print("🌐 Running in PRODUCTION mode (cloud deployment)")
    else:
        print("💻 Running in LOCAL mode")
        print("📖 Visit: http://localhost:5000")
    
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run with appropriate settings for production vs local
    debug_mode = not bool(os.environ.get('PORT'))  # Debug only in local mode
    app.run(debug=debug_mode, host='0.0.0.0', port=port)