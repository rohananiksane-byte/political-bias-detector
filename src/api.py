"""
Flask API for Political Bias Detection
Optimized for Render deployment
"""

import os
import datetime
import gc
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import with error handling for deployment
try:
    from bias_analyzer import EnhancedContextualAnalyzer as BiasAnalyzer
    ANALYZER_IMPORT_SUCCESS = True
except Exception as e:
    print(f"⚠️ Warning: Could not import analyzer at startup: {e}")
    BiasAnalyzer = None
    ANALYZER_IMPORT_SUCCESS = False

# -------------------------------------------------------------------------
# Flask App Setup
# -------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Analyzer Initialization
# -------------------------------------------------------------------------
analyzer = None

def initialize_analyzer():
    """Initialize the analyzer with error handling."""
    global analyzer
    if analyzer is not None:
        return analyzer
    
    if not ANALYZER_IMPORT_SUCCESS:
        raise RuntimeError("BiasAnalyzer could not be imported")
    
    try:
        logger.info("🚀 Initializing Political Bias Analyzer...")
        analyzer = BiasAnalyzer(verbose=False)
        logger.info("✅ Analyzer successfully loaded")
        return analyzer
    except Exception as e:
        logger.error(f"❌ Failed to initialize analyzer: {str(e)}", exc_info=True)
        raise

# Pre-load analyzer at startup (only in production)
if os.environ.get('PORT'):
    try:
        logger.info("Production mode detected - pre-loading analyzer...")
        initialize_analyzer()
    except Exception as e:
        logger.error(f"Failed to pre-load analyzer: {e}")

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
MAX_CONTENT_LENGTH = 5000  # Increased limit

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def get_analyzer():
    """Get or initialize the analyzer."""
    global analyzer
    if analyzer is None:
        logger.info("Analyzer not loaded - initializing now...")
        initialize_analyzer()
    return analyzer

# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------
@app.route('/')
def home():
    """Root endpoint - API info."""
    return jsonify({
        'name': 'Political Bias Detection API',
        'version': '1.0',
        'status': 'online',
        'analyzer_status': 'loaded' if analyzer is not None else 'not_loaded',
        'endpoints': {
            '/api/analyze': 'POST - Analyze text or URL',
            '/api/health': 'GET - Check API status',
            '/health': 'GET - Health check'
        },
        'documentation': 'https://github.com/rohananiksane-byte/political-bias-detector'
    })

@app.route('/api/health')
@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'analyzer_loaded': analyzer is not None,
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
    }), 200

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze content for political bias."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        if 'content' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "content" field in request'
            }), 400
        
        content = str(data['content']).strip()
        
        if not content:
            return jsonify({
                'success': False,
                'error': 'Content cannot be empty'
            }), 400
        
        if len(content) > MAX_CONTENT_LENGTH:
            return jsonify({
                'success': False,
                'error': f'Content too long. Maximum {MAX_CONTENT_LENGTH} characters allowed.'
            }), 400
        
        # Get analyzer
        try:
            analyzer_instance = get_analyzer()
        except Exception as e:
            logger.error(f"Failed to get analyzer: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Analyzer initialization failed. Please try again in a moment.'
            }), 503
        
        # Analyze content
        logger.info(f"Analyzing content ({len(content)} chars)...")
        result = analyzer_instance.analyze(content)
        
        # Check for errors in result
        if 'error' in result:
            logger.warning(f"Analysis returned error: {result['error']}")
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        logger.info(f"✅ Analysis complete - Score: {result.get('bias_score', 'N/A')}")
        
        # Clean up memory
        gc.collect()
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error during analysis'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': {
            '/': 'GET - API information',
            '/api/health': 'GET - Health check',
            '/health': 'GET - Health check',
            '/api/analyze': 'POST - Analyze content'
        }
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'success': False,
        'error': f'Method not allowed. Use POST for /api/analyze'
    }), 405

# -------------------------------------------------------------------------
# App Entry Point
# -------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = not bool(os.environ.get('PORT'))
    
    print("\n" + "="*70)
    print("🚀 POLITICAL BIAS DETECTION API")
    print("="*70)
    print(f"📍 Port: {port}")
    print(f"🌐 Mode: {'PRODUCTION' if os.environ.get('PORT') else 'LOCAL'}")
    print(f"🔧 Debug: {debug_mode}")
    print("="*70 + "\n")
    
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=port,
        threaded=True
    )