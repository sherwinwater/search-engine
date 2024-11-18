import argparse
import json
import shutil

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
from service.document_clustering import DocumentClustering
from service.scrape_web import HTMLScraper
from service.text_search import TextSearch
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


def delete_files_except(base_path, exceptions, task_id):
    """
    Delete all files and folders in the given path except those specified in exceptions.

    Args:
        base_path (str): Path to the directory where deletion will occur
        exceptions (list): List of file/folder names to preserve
    """
    # Ensure the path exists
    logger = setup_logging(name=f"{__name__}", task_id=task_id)

    if not os.path.exists(base_path):
        logger.info(f"Path {base_path} does not exist")
        return

    # List all items in the directory
    items = os.listdir(base_path)

    # Delete each item unless it's in exceptions
    for item in items:
        item_path = os.path.join(base_path, item)

        # Skip if item is in exceptions
        if item in exceptions:
            logger.info(f"Preserving: {item}")
            continue

        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
                logger.info(f"Deleted file: {item}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                logger.info(f"Deleted directory: {item}")
        except Exception as e:
            logger.exception(f"Error deleting {item}: {str(e)}")


def background_scraping(url, output_dir, task_id, max_workers=10, max_pages=50):
    """Background task for site analysis and downloading."""
    try:
        db = SearchEngineDatabase()
        scraper = HTMLScraper(base_url=url, output_dir=output_dir, task_id=task_id, max_pages=max_pages)
        db.update_scraping_task(task_id, scraper)

        if scraper.analyze_site(max_workers=max_workers):
            scraper.download_all(max_workers=max_workers)
    except Exception as e:
        scraper.update_status('failed', f'Error during scraping: {str(e)}')
    finally:
        db.close()


def background_building_index(docs_dir: str, task_id: str, url: str):
    """Background task for site analysis and downloading."""
    try:
        buildTextIndex = BuildTextIndex(docs_dir=docs_dir, task_id=task_id, scraping_url=url)

        buildTextIndex.load_documents()
        buildTextIndex.build_index()
        buildTextIndex.save_index()

    except Exception as e:
        logger.exception(e)


def background_clustering(docs_dir, task_id, url):
    """Background task for building clusters"""
    try:
        db = SearchEngineDatabase()
        try:
            # Update status to processing
            db.update_clustering_status(task_id, 'processing')

            # Set up paths
            webpage_graph = os.path.join(docs_dir, 'webpage_graph.json')

            # Initialize and run clustering
            cluster = DocumentClustering(task_id, docs_dir, webpage_graph)
            cluster.build_clustering_data()

            # Update status to completed
            db.update_clustering_status(task_id, 'completed')

            exceptions = ['webpage_graph.json', 'clustering_data']
            # delete_files_except(docs_dir, exceptions, task_id)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in clustering for task {task_id}: {error_msg}")
            db.update_clustering_status(task_id, 'failed', error_msg)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Critical error in clustering thread: {str(e)}")


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
        url = scraper.get('base_path')

        # Start background thread
        thread = threading.Thread(
            target=background_building_index,
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
            })

        return jsonify(status)
    finally:
        db.close()


@app.route('/api/clustering_status/<task_id>', methods=['GET'])
def get_clustering_status(task_id):
    """Get the status of a scraping task."""
    db = SearchEngineDatabase()
    try:
        index_status = db.get_building_text_index_status(task_id)
        if not index_status:
            # Check if task exists but status not yet set
            task_exists = db.get_web_scraping_status(task_id)
            if task_exists:
                return jsonify({
                    "status": "pending",
                    "message": "Task initialized, waiting for status update"
                }), 202
            return jsonify({
                "error": "Task not found"
            }), 404

        clustering_status = db.get_clustering_status(task_id)

        if clustering_status and clustering_status.get('status') == 'completed':
            return jsonify(index_status)

        return jsonify({
            "status": 'clustering'
        })

    except Exception as e:
        logger.error(f"Clustering status error for task {task_id}: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "error": str(e)
        }), 500
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
    """
    Combined endpoint to scrape URL, build index, and cluster content.
    Required parameters: url
    """
    try:
        # Get URL from request
        data = request.get_json()
        url = data.get('url')
        max_pages = data.get('max_pages', 50)

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
                current_dir = os.path.dirname(os.path.abspath(__file__))
                root_dir = os.path.dirname(current_dir)
                abs_destination_path = os.path.join(root_dir, 'scraped_data', task_id)

                os.makedirs(abs_destination_path, exist_ok=True)

                # Start scraping in background
                thread = threading.Thread(
                    target=background_scraping,
                    args=(url, abs_destination_path, task_id, 10, max_pages)
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

            # Get scraper status to get docs_dir
            scraper = db.get_web_scraping_status(task_id)
            docs_dir = scraper.get('output_dir')

            # Check and start index building if needed
            index_status = db.get_building_text_index_status(task_id)
            if not index_status or index_status.get('status') != 'completed':
                # Start building index
                thread = threading.Thread(
                    target=background_building_index,
                    args=(docs_dir, task_id, url)
                )
                thread.daemon = True
                thread.start()

                # Wait for indexing to complete
                max_wait_time = 300  # 5 minutes timeout
                start_time = time.time()

                while True:
                    index_status = db.get_building_text_index_status(task_id)
                    if index_status:
                        if index_status.get('status') == 'completed':
                            break
                        elif index_status.get('status') == 'failed':
                            return jsonify({"error": "Indexing failed", "details": index_status.get('error')}), 500

                    if time.time() - start_time > max_wait_time:
                        return jsonify({"error": "Indexing timeout"}), 504

                    time.sleep(2)

            # Check and start clustering if needed
            clustering_status = db.get_clustering_status(task_id)
            if not clustering_status or clustering_status.get('status') != 'completed':
                # Start clustering in background
                thread = threading.Thread(
                    target=background_clustering,
                    args=(docs_dir, task_id, url)
                )
                thread.daemon = True
                thread.start()

            # Return response based on overall status
            clustering_status = db.get_clustering_status(task_id)
            if clustering_status and clustering_status.get('status') == 'completed':
                status = 'completed'
            else:
                status = 'processing'

            return jsonify({
                "task_id": task_id,
                "url": url,
                "status": status,
                "scraping_status": scraper.get('status'),
                "indexing_status": index_status.get('status') if index_status else 'not_started',
                "clustering_status": clustering_status.get('status') if clustering_status else 'not_started'
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in build_index_by_url endpoint: {str(e)}")
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
        debug=False,
        allow_unsafe_werkzeug=True
    )
