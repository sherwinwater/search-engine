import argparse
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import uuid
import os
import validators
from flask_socketio import SocketIO, join_room, leave_room
from threading import Lock
from urllib.parse import urlparse
import time

from db.database import SearchEngineDatabase
from service.build_text_index import BuildTextIndex
from service.scrape_web import HTMLScraper
from service.text_search import TextSearch
from utils.convert_numpy_types import convert_numpy_types
from utils.json_serialize import prepare_scraper_status_for_json, prepare_index_status_for_json
from utils.setup_logging import setup_logging, SocketIOLogHandler

app = Flask(__name__)

# CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, resources={
    r"/*": {  # Apply to all routes under /api/
        "origins": "*",  # Allow all origins
        "methods": ["GET", "POST", "OPTIONS"],  # Allowed methods
        "allow_headers": ["Content-Type", "Authorization"],  # Allowed headers
        "supports_credentials": True,  # Allow credentials
        "expose_headers": ["Content-Range", "X-Content-Range"]  # Expose these headers to the frontend
    }
})

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8,
    manage_session=False
)

log_lock = Lock()
socket_handler = SocketIOLogHandler.init_handler(socketio)
logger = setup_logging(__name__)

active_connections = {}


@socketio.on('connect')
def handle_connect():
    sid = request.sid
    active_connections[sid] = set()  # Initialize empty set for room membership
    logger.info(f'Client connected: {sid}')


@socketio.on('join')
def handle_join(task_id):
    sid = request.sid
    try:
        join_room(task_id)
        active_connections[sid].add(task_id)
        logger.info(f'Client {sid} joined room {task_id}')
        socketio.emit('status', {'message': f'Joined room {task_id}'}, room=task_id)
    except Exception as e:
        logger.error(f'Error joining room: {str(e)}')


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in active_connections:
        # Leave all rooms this client was in
        for room in active_connections[sid]:
            leave_room(room)
            socketio.emit('status', {'message': f'Client left room {room}'}, room=room)
        del active_connections[sid]
    logger.info(f'Client disconnected: {sid}')


def emit_log(task_id, message):
    """Utility function to emit logs to a specific task room"""
    socketio.emit('log_message', {'message': message}, room=task_id)


@app.route("/health")
def home():
    logger.info("Health check OK")
    data = {
        "message": "Hello, World!",
        "status": "success"
    }
    return jsonify(data)


def background_scraping(url, output_dir, task_id, max_workers=10):
    """Background task for site analysis and downloading."""
    try:
        db = SearchEngineDatabase()
        scraper = HTMLScraper(base_url=url, output_dir=output_dir, task_id=task_id)
        db.update_scraping_task(task_id, scraper)

        if scraper.analyze_site(max_workers=max_workers):
            scraper.download_all(max_workers=max_workers)
    except Exception as e:
        scraper.update_status('failed', f'Error during scraping: {str(e)}')
    finally:
        db.close()


def background_building_index(docs_dir: str, task_id: str):
    """Background task for site analysis and downloading."""
    try:
        buildTextIndex = BuildTextIndex(docs_dir=docs_dir, task_id=task_id)

        buildTextIndex.load_documents()
        buildTextIndex.build_index()
        buildTextIndex.save_index()

    except Exception as e:
        logger.exception(e)


@app.route('/api/scrape_web', methods=['GET'])
def scrape_web():
    """Start web scraping task."""
    db = SearchEngineDatabase()
    url = request.args.get('url')

    if not url:
        return jsonify({
            "error": "URL parameter is required"
        }), 400

    # Validate URL format
    if not validators.url(url):
        return jsonify({
            "error": f"Invalid URL format: {url}"
        }), 400

    existing_task = db.get_web_scraping_data_by_url(url)
    if existing_task:
        return jsonify({
            "task_id": existing_task['task_id'],
            "message": "URL was already scraped"
        }), 200

    max_workers = int(request.args.get('max_workers', 10))

    if not url:
        return jsonify({
            "error": "URL parameter is required"
        }), 400

    task_id = str(uuid.uuid4())
    destination_path = os.path.join('scraped_data', task_id)
    os.makedirs(destination_path, exist_ok=True)

    # Start background thread
    thread = threading.Thread(
        target=background_scraping,
        args=(url, destination_path, task_id, max_workers)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        "task_id": task_id,
        "message": "Scraping started",
        "status_endpoint": f"/scrape_status/{task_id}"
    }), 202


@app.route('/api/scrape_status/<task_id>', methods=['GET'])
def get_scrape_web_status(task_id):
    """Get the status of a scraping task."""
    db = SearchEngineDatabase()
    try:
        scraper = db.get_web_scraping_status(task_id)
        if not scraper:
            return jsonify({
                "error": "Task not found"
            }), 404

        scraper = prepare_scraper_status_for_json(scraper)

        if scraper.get('status') == 'completed':
            webpage_graph_file = scraper.get('webpage_graph_file')
            if webpage_graph_file:
                with open(webpage_graph_file, 'r') as f:
                    scraper['webpage_graph'] = json.load(f)

        return jsonify(scraper)
    finally:
        db.close()


@app.route('/api/build_text_index/<task_id>', methods=['GET'])
def build_text_index(task_id):
    """Get the status of a scraping task."""
    db = SearchEngineDatabase()

    try:
        scraper = db.get_web_scraping_status(task_id)
        if not scraper:
            return jsonify({
                "error": "Task not found"
            }), 404

        docs_dir = scraper.get('output_dir')

        # Start background thread
        thread = threading.Thread(
            target=background_building_index,
            args=(docs_dir, task_id)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "task_id": task_id,
            "message": "Building text index",
            "status_endpoint": f"/build_text_index_status/{task_id}"
        }), 202
    finally:
        db.close()


@app.route('/api/build_text_index/local', methods=['GET'])
def build_text_index_with_local_data():
    """Get the status of a scraping task."""
    docs_dir = request.args.get('docs_dir')
    if not docs_dir:
        return jsonify({
            "error": "URL parameter is required"
        }), 400

    try:
        docs_dir_name = docs_dir.split("/")[-1]
        task_id = f"{docs_dir_name}_{str(uuid.uuid4())}"

        # Start background thread
        thread = threading.Thread(
            target=background_building_index,
            args=(docs_dir, task_id)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "task_id": task_id,
            "message": "Building text index",
            "status_endpoint": f"/build_text_index_status/{task_id}"
        }), 202
    except Exception as e:
        logger.error(f"Building text index error for task {task_id}: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "error": str(e)
        }), 500


@app.route('/api/build_text_index_status/<task_id>', methods=['GET'])
def get_build_text_index_status(task_id):
    """Get the status of a scraping task."""
    db = SearchEngineDatabase()
    try:
        status = db.get_building_text_index_status(task_id)
        if not status:
            return jsonify({
                "error": "Task not found"
            }), 404

        return jsonify(prepare_index_status_for_json(status))
    finally:
        db.close()


@app.route('/api/text_index_status_by_url', methods=['POST'])
def get_text_index_status_by_url():
    """Get the status of a scraping task."""
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({
            "error": "'url' is required"
        }), 400

    db = SearchEngineDatabase()
    try:
        status = db.get_build_index_by_url(url)
        return jsonify(status), 200
    finally:
        db.close()


@app.route('/api/search_text/<task_id>', methods=['GET'])
def search_text(task_id):
    """Search indexed documents for the given task."""
    try:
        # Get query parameter
        query = request.args.get('query')
        if not query:
            return jsonify({
                "status": "error",
                "message": "No query provided"
            }), 400

        # Get database connection
        db = SearchEngineDatabase()

        try:
            # Check task status
            status = db.get_building_text_index_status(task_id)
            if not status:
                return jsonify({
                    "status": "error",
                    "message": "Task not found"
                }), 404

            # Check if indexing is complete
            if status.get('status') != 'completed':
                return jsonify({
                    "status": "error",
                    "message": "Index building not completed",
                    "task_status": status.get('status')
                }), 400

            # Initialize searcher
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            index_path = os.path.join(parent_dir, 'index_data', f"{task_id}.pkl")
            searcher = TextSearch(index_path=index_path)
            results = searcher.search(query, top_k=5)
            return jsonify(convert_numpy_types(results))

        finally:
            db.close()

    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": "Index not found",
            "error": str(e)
        }), 404

    except Exception as e:
        logger.error(f"Search error for task {task_id}: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "error": str(e)
        }), 500


@app.route('/api/build_index_by_url', methods=['POST'])
def build_index_by_url():
    """
    Combined endpoint to scrape URL, build index, and search content.
    Required parameters: url, query
    """
    try:
        # Get URL and search query from request
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({
                "error": "'url' is required"
            }), 400

        # Validate URL format
        if not validators.url(url):
            return jsonify({
                "error": f"Invalid URL format: {url}"
            }), 400

        # Extract domain from URL
        domain = urlparse(url).netloc

        # Initialize database connection
        db = SearchEngineDatabase()
        try:
            # Check if domain has been scraped before
            existing_tasks = db.get_web_scraping_data_by_url(url)
            task_id = None

            if existing_tasks:
                # Use existing scrape data
                task_id = existing_tasks['task_id']
                logger.info(f"Using existing scrape data for domain {domain} with task_id: {task_id}")
            else:
                # Start new scraping task
                task_id = str(uuid.uuid4())
                destination_path = os.path.join('scraped_data', task_id)
                os.makedirs(destination_path, exist_ok=True)

                # Start scraping in background
                thread = threading.Thread(
                    target=background_scraping,
                    args=(url, destination_path, task_id, 10)
                )
                thread.daemon = True
                thread.start()

                # Wait for scraping to complete
                max_wait_time = 300  # 5 minutes timeout
                start_time = time.time()

                while True:
                    scraper = db.get_web_scraping_status(task_id)
                    if scraper:
                        if scraper.get('status') == 'completed':
                            break
                        elif scraper.get('status') == 'failed':
                            return jsonify({"error": "Scraping failed", "details": scraper.get('error')}), 500

                    if time.time() - start_time > max_wait_time:
                        return jsonify({"error": "Scraping timeout"}), 504

                    time.sleep(2)

            # Start index building if needed
            index_status = db.get_building_text_index_status(task_id)
            if not index_status or index_status.get('status') != 'completed':
                # Get scraper status to get docs_dir
                scraper = db.get_web_scraping_status(task_id)
                docs_dir = scraper.get('output_dir')

                # Start building index
                thread = threading.Thread(
                    target=background_building_index,
                    args=(docs_dir, task_id)
                )
                thread.daemon = True
                thread.start()

            else:
                logger.info(f"Using existing index for domain {domain} with task_id: {task_id}")
                return jsonify({
                    "task_id": task_id,
                    "url": url,
                    "status": 'completed',
                })

            return jsonify({
                "task_id": task_id,
                "url": url,
                "status": 'processing',
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in search_url endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "error": str(e)
        }), 500


@app.route('/api/search_url', methods=['POST'])
def search_url():
    """
    Combined endpoint to scrape URL, build index, and search content.
    Required parameters in request body: url, query
    """
    try:
        # Get data from request body
        data = request.get_json()
        url = data.get('url')
        search_query = data.get('query')

        if not url or not search_query:
            return jsonify({
                'error': 'Missing required parameters: url and query must be provided'
            }), 400

        # Validate URL format
        if not validators.url(url):
            return jsonify({
                "error": f"Invalid URL format: {url}"
            }), 400

        domain = urlparse(url).netloc

        db = SearchEngineDatabase()
        try:
            existing_tasks = db.get_web_scraping_data_by_url(url)

            if not existing_tasks:
                return jsonify({
                    "error": f"Domain {domain} has not been scraped yet"
                }), 404

            task_id = existing_tasks['task_id']

            index_status = db.get_building_text_index_status(task_id)
            if not index_status or index_status.get('status') != 'completed':
                return jsonify({
                    "error": f"Index for domain {domain} has not been built yet"
                }), 404

            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            index_path = os.path.join(parent_dir, 'index_data', f"{task_id}.pkl")
            searcher = TextSearch(index_path=index_path)
            results = searcher.search(search_query, top_k=5)

            return jsonify({
                "task_id": task_id,
                "url": url,
                "query": search_query,
                "results": convert_numpy_types(results)
            }), 200

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in search_url endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5009)
    args = parser.parse_args()

    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=False)
