import argparse
import json
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import uuid
import os
import validators
from flask_socketio import SocketIO, join_room, leave_room
from threading import Lock
from urllib.parse import urlparse
from datetime import datetime

from db.database import SearchEngineDatabase
from service.text_search import TextSearch
from service.thread_manager import ThreadManager
from utils.convert_numpy_types import convert_numpy_types
from utils.json_serialize import prepare_scraper_status_for_json
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

flask_env = os.getenv('FLASK_ENV', 'development')
async_mode = 'eventlet' if flask_env == 'production' else 'threading'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode=async_mode,  # threading for development, useing eventlet for production
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

thread_manager = ThreadManager()

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
        target=thread_manager.start_background_scraping,
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
        url = scraper.get('base_path')

        # Start background thread
        thread = threading.Thread(
            target=thread_manager.start_background_building_text_index,
            args=(docs_dir, task_id, url)
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


@app.route('/api/text_indexes', methods=['GET'])
def get_all_building_text_index():
    """Get the status of a scraping task."""
    db = SearchEngineDatabase()

    try:
        text_indexes = db.get_all_building_text_index()
        return jsonify(text_indexes), 200
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
            target=thread_manager.start_background_building_text_index,
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
            })

        return jsonify(status)
    finally:
        db.close()


@app.route('/api/clustering_status/<task_id>', methods=['GET'])
def get_clustering_status(task_id):
    db = SearchEngineDatabase()
    try:
        status_info = {
            "task_id": task_id,
            "scraping_url": "",
            "processed_files": "",
            "created_at": ""
        }

        index_status = db.get_building_text_index_status(task_id)
        if not index_status:
            task_exists = db.get_web_scraping_status(task_id)
            if not task_exists:
                return jsonify({"error": "Task not found"}), 404

            status_info['scraping_url'] = task_exists.get('scraping_url')
            status_info['processed_files'] = task_exists.get('processed_files')
            status_info['created_at'] = datetime.fromtimestamp(task_exists.get('start_time'))

            if task_exists.get('status') != 'completed':
                status_info["status"] = "downloading"
            else:
                status_info["status"] = "indexing"

            return jsonify(status_info)

        status_info["status"] = "indexing"
        status_info['scraping_url'] = index_status.get('scraping_url')
        status_info['created_at'] = index_status.get('created_at')

        processed_files = index_status.get('processed_files')
        if processed_files == 0:
            scraping_data = db.get_web_scraping_status(task_id)
            processed_files = scraping_data.get('processed_files')

        status_info['processed_files'] = processed_files

        if index_status.get('status') != 'completed':
            return jsonify(status_info)

        clustering_status = db.get_clustering_status(task_id)
        status_info["status"] = "completed" if clustering_status and clustering_status.get('status') == 'completed' else "clustering"

        return jsonify(status_info)

    except Exception as e:
        logger.error(f"Clustering status error for task {task_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/text_index/<task_id>', methods=['GET'])
def get_build_text_index_by_task_id(task_id):
    db = SearchEngineDatabase()
    try:
        text_index = db.get_building_text_index_status(task_id)
        if not text_index:
            return jsonify({
                "error": "Task not found"
            }), 404

        parent_dir = text_index.get('docs_dir')

        # Add webpage graph data
        webpage_graph_file = os.path.join(parent_dir, 'webpage_graph.json')
        if os.path.exists(webpage_graph_file):
            with open(webpage_graph_file, 'r') as f:
                text_index['webpage_graph'] = json.load(f)

        # Add clustering data
        clustering_status = db.get_clustering_status(task_id)
        if clustering_status:
            text_index['clustering_status'] = clustering_status

            # If clustering is completed, add the cluster structure
            if clustering_status.get('status') == 'completed':
                cluster_structure_file = os.path.join(parent_dir, 'clustering_data', 'cluster_structure.json')
                if os.path.exists(cluster_structure_file):
                    try:
                        with open(cluster_structure_file, 'r') as f:
                            text_index['clustering_data'] = json.load(f)
                    except Exception as e:
                        logger.error(f"Error reading cluster structure: {e}")
                        text_index['clustering_data'] = None

        return jsonify(text_index)
    except Exception as e:
        logger.error(f"Error in get_build_text_index_by_task_id: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500
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
            results = searcher.search_with_suggestions(query)
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
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        max_pages = data.get('max_pages', 40)

        if not url or url == '' or not validators.url(url):
            return jsonify({"error": "Invalid or missing URL"}), 400

        db = SearchEngineDatabase()
        try:
            # Check existing tasks
            existing_tasks = db.get_web_scraping_data_by_url(url)
            if existing_tasks:
                task_id = existing_tasks['task_id']
            else:
                # Initialize new task
                task_id = str(uuid.uuid4())
                current_dir = os.path.dirname(os.path.abspath(__file__))
                abs_destination_path = os.path.join(current_dir, '..', 'scraped_data', task_id)
                os.makedirs(abs_destination_path, exist_ok=True)

                # Start background processing
                # thread = threading.Thread(
                #     target=thread_manager.process_url_pipeline,
                #     args=(url, abs_destination_path, task_id, max_pages)
                # )
                # thread.daemon = True
                # thread.start()

                thread_manager.start_pipeline(url, abs_destination_path, task_id, max_pages)

            # Return immediately with status
            return jsonify({
                "task_id": task_id,
                "url": url,
                "status": "processing",
                "message": "Processing started"
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error initiating processing: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route('/api/kill/<task_id>', methods=['POST'])
def kill_process(task_id: str):
    try:
        if not task_id:
            return jsonify({"error": "Task ID is required"}), 400

        def cleanup_sequence(task_id: str):
            try:
                # Start cancellation
                cancellation_success = thread_manager.start_cancellation(task_id)

                if not cancellation_success:
                    logger.warning(f"No active processing found for task {task_id}")
                    return

                # Wait for 10 seconds
                time.sleep(10)

                # Start cleanup process
                thread_manager.start_cleanup(task_id)

            except Exception as e:
                logger.error(f"Error in cleanup sequence for {task_id}: {str(e)}")

        # Start the cleanup sequence in a separate thread
        cleanup_thread = threading.Thread(
            target=cleanup_sequence,
            args=(task_id,)
        )
        cleanup_thread.start()

        return jsonify({
            "task_id": task_id,
            "status": "cancelling",
            "message": "Cancellation and cleanup sequence initiated"
        })

    except Exception as e:
        logger.error(f"Error initiating cancellation and cleanup: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
@app.route('/api/delete/<task_id>', methods=['POST'])
def delete_task(task_id: str):
    try:
        if not task_id:
            return jsonify({"error": "Task ID is required"}), 400

        def cleanup_sequence(task_id: str):
            try:
                thread_manager.start_cleanup(task_id)
            except Exception as e:
                logger.error(f"Error in cleanup sequence for {task_id}: {str(e)}")

        # Start the cleanup sequence in a separate thread
        cleanup_thread = threading.Thread(
            target=cleanup_sequence,
            args=(task_id,)
        )
        cleanup_thread.start()

        return jsonify({
            "task_id": task_id,
            "status": "deleting",
            "message": "Deleting all data related to task"
        })

    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


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
            results = searcher.search_with_suggestions(search_query)
            return jsonify(convert_numpy_types(results))

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
        debug=False,
        allow_unsafe_werkzeug=True
    )
