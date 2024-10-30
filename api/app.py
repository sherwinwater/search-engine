from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import uuid
import os
import validators

from db.database import SearchEngineDatabase
from service.build_text_index import BuildTextIndex
from service.scrape_web import HTMLScraper
from service.text_search import TextSearch
from utils.convert_numpy_types import convert_numpy_types
from utils.json_serialize import prepare_scraper_status_for_json, prepare_index_status_for_json
from utils.setup_logging import setup_logging

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

logger = setup_logging(__name__)
@app.route("/health")
def home():
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
    url = request.args.get('url')
    # Match existing error format exactly
    print(url)
    if not url:
        return jsonify({
            "error": "URL parameter is required"
        }), 400

    # Validate URL format
    if not validators.url(url):
        return jsonify({
            "error": f"Invalid URL format: {url}"
        }), 400

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
