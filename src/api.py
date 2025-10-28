"""
Flask API for Political Bias Detection
Memory-optimized for Render (512MB free instance)
"""

import os
import datetime
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
from bias_analyzer import EnhancedContextualAnalyzer as BiasAnalyzer
import logging

app = Flask(__name__)

# Enable CORS for all origins
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Analyzer global (lazy-loaded)
analyzer = None

# Max allowed content length to prevent memory blowup
MAX_CONTENT_LENGTH = 3000  # characters

def get_analyzer():
    """Lazy-load the analyzer to save memory."""
    global analyzer
    if analyzer is None:
        logger.info("Initializing analyzer...")
        try:
            # Optionally, switch to a smaller model to reduce memory
            # analyzer = BiasAnalyzer(model_name='distilbert-base-uncased-finetuned-sst-2', verbose=False)
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
@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': str(datetime.datetime.now()),
        'analyzer_loaded': analyzer is not None
    })

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze content for political bias."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'success': False, 'error': 'Missing content field'}), 400

        content = data['content'].strip()
        if len(content) == 0:
            return jsonify({'success': False, 'error': 'Content cannot be empty'}), 400

        # Enforce max content length
        if len(content) > MAX_CONTENT_LENGTH:
            return jsonify({
                'success': False,
                'error': f'Content too long ({len(content)} chars). Max allowed is {MAX_CONTENT_LENGTH}.'
            }), 400

        # Lazy-load analyzer
        analyzer_instance = get_analyzer()

        # Run analysis
        logger.info(f"Analyzing content (length: {len(content)} chars)")
        result = analyzer_instance.analyze(content)

        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 400

        logger.info(f"Analysis complete - Score: {result.get('bias_score')}")

        # Clean up memory
        gc.collect()

        return jsonify({'success': True, 'data': result}), 200

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'An error occurred during analysis. Please try again.'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/api/analyze', '/api/health', '/health', '/']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = not bool(os.environ.get('PORT'))

    print("\n" + "="*60)
    print("🚀 STARTING BIAS DETECTION API")
    print("="*60)
    print(f"📍 API running on port: {port}")
    print("🌐 PRODUCTION mode" if os.environ.get('PORT') else "💻 LOCAL mode")
    print("="*60 + "\n")

    app.run(debug=debug_mode, host='0.0.0.0', port=port)