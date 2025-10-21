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
from bias_analyzer import EnhancedContextualAnalyzer as BiasAnalyzer

# -------------------------------------------------------------------------
# Flask App Setup
# -------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Analyzer Initialization (Preload once at startup)
# -------------------------------------------------------------------------
print("🚀 Initializing Political Bias Analyzer...")

try:
    analyzer = BiasAnalyzer(verbose=False)
    print("✅ Analyzer successfully loaded at startup.")
except Exception as e:
    analyzer = None
    print("❌ Failed to load analyzer at startup:", e)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
MAX_CONTENT_LENGTH = 3000  # limit text size to prevent memory overuse

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def get_analyzer():
    """Lazy-load or reuse the analyzer."""
    global analyzer
    if analyzer is None:
        logger.info("Analyzer not loaded — attempting lazy initialization...")
        try:
            analyzer = BiasAnalyzer(verbose=False)
            logger.info("✅ Analyzer initialized successfully (lazy-load).")
        except Exception as e:
            logger.error(f"❌ Failed to initialize analyzer lazily: {str(e)}")
            raise
    return analyzer

# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------
@app.route('/')
def home():
    """Root endpoint — shows API info."""
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
        'analyzer_loaded': analyzer is not None,
        'timestamp': str(datetime.datetime.utcnow())
    })

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze content for political bias."""
    if request.method == 'OPTIONS':
        return '', 204  # preflight CORS

    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'success': False, 'error': 'Missing content field'}), 400

        content = data['content'].strip()
        if not content:
            return jsonify({'success': False, 'error': 'Content cannot be empty'}), 400

        if len(content) > MAX_CONTENT_LENGTH:
            return jsonify({
                'success': False,
                'error': f'Content too long ({len(content)} chars). Max allowed is {MAX_CONTENT_LENGTH}.'
            }), 400

        analyzer_instance = get_analyzer()
        logger.info(f"Analyzing content (length: {len(content)} chars)...")

        result = analyzer_instance.analyze(content)

        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 400

        logger.info(f"✅ Analysis complete - Bias Score: {result.get('bias_score')}")

        gc.collect()  # memory cleanup
        return jsonify({'success': True, 'data': result}), 200

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal error during analysis'}), 500

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

# -------------------------------------------------------------------------
# App Entry Point
# -------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = not bool(os.environ.get('PORT'))

    print("\n" + "="*60)
    print("🚀 STARTING POLITICAL BIAS DETECTION API")
    print("="*60)
    print(f"📍 Running on port: {port}")
    print("🌐 PRODUCTION mode" if os.environ.get('PORT') else "💻 LOCAL mode")
    print("="*60 + "\n")

    app.run(debug=debug_mode, host='0.0.0.0', port=port)