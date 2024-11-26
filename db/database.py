import os
import sqlite3
import json
import time
from datetime import datetime
from threading import local
from urllib.parse import urlparse

from utils.setup_logging import setup_logging

logger = setup_logging(__name__)


class ThreadSafeDB:
    def __init__(self, db_name):
        self.db_name = db_name
        self._local = local()

    def get_connection(self):
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_name)
        return self._local.connection

    def get_cursor(self):
        if not hasattr(self._local, 'cursor'):
            self._local.cursor = self.get_connection().cursor()
        return self._local.cursor

    def close(self):
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection
        if hasattr(self._local, 'cursor'):
            del self._local.cursor


class SearchEngineDatabase:
    def __init__(self, db_name='search_engine.db'):
        self.db_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_name)
        logger.info(f"Database path: {self.db_name}")
        self.thread_safe_db = ThreadSafeDB(self.db_name)
        self.setup_database()

    @property
    def conn(self):
        return self.thread_safe_db.get_connection()

    @property
    def cursor(self):
        return self.thread_safe_db.get_cursor()

    def setup_database(self):
        """Initialize database and create tables"""
        try:
            # Create tasks table with additional fields
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    domain TEXT,
                    base_path TEXT,
                    output_dir TEXT,
                    visited_urls TEXT,  -- Stored as JSON array
                    page_sizes TEXT,    -- Stored as JSON array
                    request_times TEXT, -- Stored as JSON array
                    downloaded_files TEXT, -- Stored as JSON array
                    downloaded_urls TEXT,  -- Stored as JSON array
                    status TEXT,
                    message TEXT,
                    webpage_graph_file TEXT,
                    start_time REAL,
                    is_completed BOOLEAN,
                    completion_time REAL
                )
            ''')

            # Create analysis table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis (
                    task_id TEXT PRIMARY KEY,
                    pages_found INTEGER,
                    urls_in_queue INTEGER,
                    average_page_size REAL,
                    total_size_kb REAL,
                    average_request_time REAL,
                    start_time REAL,
                    completion_time REAL,
                    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
                )
            ''')

            # Create download table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS downloads (
                    task_id TEXT PRIMARY KEY,
                    total_pages INTEGER,
                    downloaded_pages INTEGER,
                    successful_downloads INTEGER,
                    failed_downloads INTEGER,
                    skipped_downloads INTEGER,
                    current_batch TEXT,
                    failed_urls TEXT,
                    progress_percentage REAL,
                    start_time REAL,
                    completion_time REAL,
                    is_completed BOOLEAN,
                    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
                )
            ''')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS text_index (
                    task_id TEXT PRIMARY KEY,
                    docs_dir TEXT NOT NULL,
                    index_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    total_files INTEGER DEFAULT 0,
                    processed_files INTEGER DEFAULT 0,
                    failed_files INTEGER DEFAULT 0,
                    progress_percentage REAL DEFAULT 0,
                    current_file TEXT,
                    start_time REAL,
                    completion_time REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    is_completed BOOLEAN DEFAULT FALSE,
                    error TEXT,
                    scraping_url TEXT
                )
            ''')

            self.cursor.execute('''
                        CREATE TABLE IF NOT EXISTS clustering (
                            task_id TEXT PRIMARY KEY,
                            status TEXT,  -- 'processing', 'completed', 'failed'
                            num_clusters INTEGER,
                            cluster_structure_path TEXT,
                            metadata_path TEXT,
                            model_path TEXT,
                            summary_path TEXT,
                            error_message TEXT,
                            start_time REAL,
                            completion_time REAL,
                            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                        )
                    ''')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS cluster_details (
                    cluster_id TEXT,
                    task_id TEXT,
                    cluster_name TEXT,
                    size INTEGER,
                    keywords TEXT,  -- Stored as JSON array
                    document_paths TEXT,  -- Stored as JSON array
                    document_urls TEXT,   -- Stored as JSON array
                    PRIMARY KEY (cluster_id, task_id),
                    FOREIGN KEY (task_id) REFERENCES clustering(task_id)
                )
            ''')

            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database setup error: {e}")
            raise

    def update_scraping_task(self, task_id, scraper):
        """Update task status with all scraper data"""
        try:
            # Get a new connection for this thread if needed
            conn = self.thread_safe_db.get_connection()
            cursor = self.thread_safe_db.get_cursor()

            # Convert sets and lists to JSON strings
            visited_urls = json.dumps(list(scraper.visited_urls))
            page_sizes = json.dumps(scraper.page_sizes)
            request_times = json.dumps(scraper.request_times)
            downloaded_files = json.dumps(scraper.downloaded_files)
            downloaded_urls = json.dumps(list(scraper.downloaded_urls))

            # Insert/update main task with all fields
            cursor.execute('''
                INSERT OR REPLACE INTO tasks (
                    task_id, domain, base_path, output_dir,
                    visited_urls, page_sizes, request_times,
                    downloaded_files, downloaded_urls,
                    status, message, start_time, is_completed, completion_time,webpage_graph_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
            ''', (
                task_id,
                scraper.domain,
                scraper.base_path,
                scraper.output_dir,
                visited_urls,
                page_sizes,
                request_times,
                downloaded_files,
                downloaded_urls,
                scraper.status['status'],
                scraper.status['message'],
                scraper.status['start_time'],
                scraper.status['is_completed'],
                scraper.status['completion_time'],
                scraper.status['webpage_graph_file']
            ))

            # Insert/update analysis
            analysis = scraper.status['analysis']
            cursor.execute('''
                INSERT OR REPLACE INTO analysis (
                    task_id, pages_found, urls_in_queue, average_page_size,
                    total_size_kb, average_request_time, start_time, completion_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_id,
                analysis['pages_found'],
                analysis['urls_in_queue'],
                analysis.get('average_page_size', 0),
                analysis.get('total_size_kb', 0),
                analysis.get('average_request_time', 0),
                analysis.get('start_time'),
                analysis.get('completion_time')
            ))

            # Insert/update download
            download = scraper.status['download']
            cursor.execute('''
                INSERT OR REPLACE INTO downloads (
                    task_id, total_pages, downloaded_pages, successful_downloads,
                    failed_downloads, skipped_downloads, current_batch, failed_urls,
                    progress_percentage, start_time, completion_time, is_completed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_id,
                download['total_pages'],
                download['downloaded_pages'],
                download['successful_downloads'],
                download['failed_downloads'],
                download['skipped_downloads'],
                json.dumps(download['current_batch']),
                json.dumps(download['failed_urls']),
                download['progress_percentage'],
                download.get('start_time'),
                download.get('completion_time'),
                download.get('is_completed', False)
            ))

            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            return False

    def get_web_scraping_data_by_url(self, url):
        """Get complete task status including all scraper data"""
        try:
            # Get a new connection for this thread if needed
            cursor = self.thread_safe_db.get_cursor()
            parsed_url = urlparse(url)
            domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # Get task data
            cursor.execute('''
                SELECT * FROM tasks 
                WHERE domain = ? OR ? LIKE '%' || domain || '%'
            ''', (domain, url))

            task_data = cursor.fetchone()

            if task_data:
                return {
                    'task_id': task_data[0],
                    'exists': True
                }
            return None
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return None

    def get_web_scraping_status(self, task_id):
        """Get complete task status including all scraper data"""
        try:
            # Get a new connection for this thread if needed
            cursor = self.thread_safe_db.get_cursor()

            # Get all raw data first
            cursor.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
            task_row = cursor.fetchone()
            if not task_row:
                return None

            # Convert row to dictionary but handle JSON fields specially
            result = {
                'task_id': task_row[0],
                'domain': task_row[1],
                'base_path': task_row[2],
                'output_dir': task_row[3],
                'visited_urls': set(json.loads(task_row[4])) if task_row[4] else set(),
                'page_sizes': json.loads(task_row[5]) if task_row[5] else {},
                'request_times': json.loads(task_row[6]) if task_row[6] else {},
                'downloaded_files': json.loads(task_row[7]) if task_row[7] else [],
                'downloaded_urls': set(json.loads(task_row[8])) if task_row[8] else set(),
                'status': task_row[9],
                'message': task_row[10],
                'webpage_graph_file': task_row[11],
                'start_time': task_row[12],
                'is_completed': bool(task_row[13]),
                'completion_time': task_row[14],
                'scraping_url': task_row[2],
                'processed_files':  len(set(json.loads(task_row[4])) if task_row[4] else set()),
            }

            # Get analysis data
            cursor.execute('SELECT * FROM analysis WHERE task_id = ?', (task_id,))
            analysis_row = cursor.fetchone()
            if analysis_row:
                result['analysis'] = {
                    'pages_found': analysis_row[1],
                    'urls_in_queue': analysis_row[2],
                    'average_page_size': analysis_row[3],
                    'total_size_kb': analysis_row[4],
                    'average_request_time': analysis_row[5],
                    'start_time': analysis_row[6],
                    'completion_time': analysis_row[7]
                }

            # Get download data
            cursor.execute('SELECT * FROM downloads WHERE task_id = ?', (task_id,))
            download_row = cursor.fetchone()
            if download_row:
                result['download'] = {
                    'total_pages': download_row[1],
                    'downloaded_pages': download_row[2],
                    'successful_downloads': download_row[3],
                    'failed_downloads': download_row[4],
                    'skipped_downloads': download_row[5],
                    'current_batch': json.loads(download_row[6]) if download_row[6] else [],
                    'failed_urls': json.loads(download_row[7]) if download_row[7] else [],
                    'progress_percentage': download_row[8],
                    'start_time': download_row[9],
                    'completion_time': download_row[10],
                    'is_completed': bool(download_row[11])
                }

            return result

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return None

    def get_all_web_scraping(self):
        try:
            self.cursor.execute('SELECT * FROM tasks')
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return []

    def update_text_index(self, task_id: str, text_index):
        """Update text index status in database"""
        try:
            conn = self.thread_safe_db.get_connection()
            cursor = self.thread_safe_db.get_cursor()

            current_time = datetime.now()

            cursor.execute('''
                INSERT OR REPLACE INTO text_index (
                    task_id, docs_dir, index_path, status, message,
                    total_files, processed_files, failed_files,
                    progress_percentage, current_file, start_time,
                    completion_time, created_at, updated_at,
                    is_completed, error,scraping_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
            ''', (
                task_id,
                text_index.docs_dir,
                text_index.index_path,
                getattr(text_index, 'status', 'pending'),
                getattr(text_index, 'message', ''),
                getattr(text_index, 'total_files', 0),
                getattr(text_index, 'processed_files', 0),
                getattr(text_index, 'failed_files', 0),
                getattr(text_index, 'progress_percentage', 0),
                getattr(text_index, 'current_file', ''),
                getattr(text_index, 'start_time', current_time),
                getattr(text_index, 'completion_time', None),
                current_time,
                current_time,
                getattr(text_index, 'is_completed', False),
                getattr(text_index, 'error', None),
                getattr(text_index, 'scraping_url', '')
            ))

            conn.commit()
            return True

        except sqlite3.Error as e:
            logger.error(f"Error updating building status: {e}")
            if conn:
                conn.rollback()
            return False

    def get_building_text_index_status(self, task_id: str):
        """Get text index building status"""
        try:
            self.cursor.execute('''
                SELECT task_id, docs_dir, index_path, status, message,
                       total_files, processed_files, failed_files,
                       progress_percentage, current_file, start_time,
                       completion_time, created_at, updated_at,
                       is_completed, error,scraping_url
                FROM text_index 
                WHERE task_id = ?
            ''', (task_id,))

            result = self.cursor.fetchone()
            if result:
                return {
                    'task_id': result[0],
                    'docs_dir': result[1],
                    'index_path': result[2],
                    'status': result[3],
                    'message': result[4],
                    'total_files': result[5],
                    'processed_files': result[6],
                    'failed_files': result[7],
                    'progress_percentage': result[8],
                    'current_file': result[9],
                    'start_time': result[10],
                    'completion_time': result[11],
                    'created_at': result[12],
                    'updated_at': result[13],
                    'is_completed': bool(result[14]),
                    'error': result[15],
                    'scraping_url': result[16]
                }
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving building status: {e}")
            return None

    def get_all_building_text_index(self):
        """Get all text index building statuses

        Returns:
            list: List of dictionaries containing text index data with named fields
        """
        try:
            self.cursor.execute('''
                SELECT task_id, docs_dir, index_path, status, message,
                       total_files, processed_files, failed_files,
                       progress_percentage, current_file, start_time,
                       completion_time, created_at, updated_at,
                       is_completed, error,scraping_url
                FROM text_index 
                order by created_at desc
            ''')

            results = self.cursor.fetchall()

            # Convert results to list of dictionaries with named fields
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'task_id': row[0],
                    'docs_dir': row[1],
                    'index_path': row[2],
                    'status': row[3],
                    'message': row[4],
                    'total_files': row[5],
                    'processed_files': row[6],
                    'failed_files': row[7],
                    'progress_percentage': row[8],
                    'current_file': row[9],
                    'start_time': row[10],
                    'completion_time': row[11],
                    'created_at': row[12],
                    'updated_at': row[13],
                    'is_completed': row[14],
                    'error': row[15],
                    'scraping_url': row[16]
                })

            return formatted_results
        except sqlite3.Error as e:
            logger.error(f"Error retrieving building status: {e}")
            return []

    def get_build_index_by_url(self, url: str):
        """
        Get text index building status by URL/domain.
        First fetches scraping task data using the domain, then retrieves text index status.

        Args:
            url (str): The URL/domain to look up

        Returns:
            dict: Combined dictionary containing scraping task and text index data
                  Returns None if no data found for the given URL
        """
        try:
            # First get the scraping task data by URL
            scraping_data = self.get_web_scraping_data_by_url(url)

            if not scraping_data:
                logger.info(f"No scraping task found for URL: {url}")
                return None

            # Get the task ID from scraping data
            task_id = scraping_data['task_id']

            # Get text index status using the task ID
            index_status = self.get_building_text_index_status(task_id)

            if not index_status:
                logger.info(f"No text index found for task ID: {task_id}")
                return {
                    'task_id': task_id,
                    'scraping_exists': True,
                    'index_exists': False,
                    'index_status': None
                }

            # Combine the data
            return {
                'task_id': task_id,
                'scraping_exists': True,
                'index_exists': True,
                'status': index_status['status'],
                'index_status': {
                    'status': index_status['status'],
                    'message': index_status['message'],
                    'progress': {
                        'total_files': index_status['total_files'],
                        'processed_files': index_status['processed_files'],
                        'failed_files': index_status['failed_files'],
                        'progress_percentage': index_status['progress_percentage'],
                        'current_file': index_status['current_file']
                    },
                    'timing': {
                        'start_time': index_status['start_time'],
                        'completion_time': index_status['completion_time'],
                        'created_at': index_status['created_at'],
                        'updated_at': index_status['updated_at']
                    },
                    'is_completed': index_status['is_completed'],
                    'error': index_status['error'],
                    'paths': {
                        'docs_dir': index_status['docs_dir'],
                        'index_path': index_status['index_path']
                    }
                }
            }

        except sqlite3.Error as e:
            logger.error(f"Database error while getting build index by URL: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while getting build index by URL: {e}")
            return None

    def start_clustering(self, task_id):
        """Record the start of clustering process"""
        try:
            current_time = time.time()
            self.cursor.execute('''
                INSERT INTO clustering (
                    task_id, status, start_time
                ) VALUES (?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status = ?,
                    start_time = ?,
                    completion_time = NULL,
                    error_message = NULL
            ''', (task_id, 'processing', current_time, 'processing', current_time))
            self.conn.commit()
        except Exception as e:
            print(f"Error starting clustering: {e}")
            raise

    def update_clustering_status(self, task_id, status, error_message=None):
        """Update clustering status and completion time"""
        try:
            current_time = time.time()
            if status == 'completed':
                self.cursor.execute('''
                    UPDATE clustering 
                    SET status = ?, completion_time = ?, error_message = ?
                    WHERE task_id = ?
                ''', (status, current_time, error_message, task_id))
            else:
                self.cursor.execute('''
                    UPDATE clustering 
                    SET status = ?, error_message = ?
                    WHERE task_id = ?
                ''', (status, error_message, task_id))
            self.conn.commit()
        except Exception as e:
            print(f"Error updating clustering status: {e}")
            raise

    def save_cluster_details(self, task_id, cluster_structure):
        """Save individual cluster details to database"""
        try:
            # First, delete any existing cluster details for this task
            self.cursor.execute('''
                DELETE FROM cluster_details WHERE task_id = ?
            ''', (task_id,))

            # Insert new cluster details
            for cluster_name, cluster_data in cluster_structure.items():
                cluster_id = f"{task_id}_{cluster_data['cluster_id']}"

                # Convert document lists to JSON strings
                document_paths = json.dumps([doc['file_path'] for doc in cluster_data['documents']])
                document_urls = json.dumps([doc['url'] for doc in cluster_data['documents']])
                keywords = json.dumps(cluster_data['keywords'])

                self.cursor.execute('''
                    INSERT INTO cluster_details (
                        cluster_id, task_id, cluster_name, size, 
                        keywords, document_paths, document_urls
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cluster_id, task_id, cluster_name, cluster_data['size'],
                    keywords, document_paths, document_urls
                ))

            self.conn.commit()
        except Exception as e:
            print(f"Error saving cluster details: {e}")
            raise

    def save_clustering_paths(self, task_id, structure_path, metadata_path, model_path, summary_path, num_clusters):
        """Save paths to clustering output files"""
        try:
            # First, make sure there's a record in the clustering table
            self.cursor.execute('''
                INSERT INTO clustering (task_id, status)
                VALUES (?, 'processing')
                ON CONFLICT(task_id) DO NOTHING
            ''', (task_id,))

            # Then update the paths
            self.cursor.execute('''
                UPDATE clustering 
                SET cluster_structure_path = ?,
                    metadata_path = ?,
                    model_path = ?,
                    summary_path = ?,
                    num_clusters = ?
                WHERE task_id = ?
            ''', (structure_path, metadata_path, model_path, summary_path, num_clusters, task_id))
            self.conn.commit()
        except Exception as e:
            print(f"Error saving clustering paths: {e}")
            raise

    def get_clustering_status(self, task_id):
        """Get current clustering status"""
        try:
            self.cursor.execute('''
                SELECT status, error_message, start_time, completion_time
                FROM clustering
                WHERE task_id = ?
            ''', (task_id,))
            result = self.cursor.fetchone()
            if result:
                return {
                    'status': result[0],
                    'error_message': result[1],
                    'start_time': result[2],
                    'completion_time': result[3]
                }
            return None
        except Exception as e:
            print(f"Error getting clustering status: {e}")
            raise

    def get_clustering_info(self, task_id):
        """Get all clustering information for a task"""
        try:
            # Get main clustering info
            self.cursor.execute('''
                SELECT * FROM clustering WHERE task_id = ?
            ''', (task_id,))
            clustering = self.cursor.fetchone()

            if not clustering:
                return None

            # Get cluster details
            self.cursor.execute('''
                SELECT * FROM cluster_details WHERE task_id = ?
            ''', (task_id,))
            clusters = self.cursor.fetchall()

            # Convert to dictionary
            clustering_dict = {
                "task_id": clustering[0],
                "status": clustering[1],
                "num_clusters": clustering[2],
                "file_paths": {
                    "structure": clustering[3],
                    "metadata": clustering[4],
                    "model": clustering[5],
                    "summary": clustering[6]
                },
                "error_message": clustering[7],
                "start_time": clustering[8],
                "completion_time": clustering[9],
                "clusters": [
                    {
                        "cluster_id": c[0],
                        "cluster_name": c[2],
                        "size": c[3],
                        "keywords": json.loads(c[4]),
                        "document_paths": json.loads(c[5]),
                        "document_urls": json.loads(c[6])
                    }
                    for c in clusters
                ]
            }

            return clustering_dict
        except Exception as e:
            print(f"Error getting clustering info: {e}")
            raise

    def delete_web_scraping_index_data(self, task_id: str) -> bool:
        """Delete all data associated with a task from all tables"""
        try:
            conn = self.thread_safe_db.get_connection()
            cursor = self.thread_safe_db.get_cursor()

            # Start transaction
            cursor.execute('BEGIN TRANSACTION')

            try:
                # Delete from tasks table
                cursor.execute('DELETE FROM tasks WHERE task_id = ?', (task_id,))

                # Delete from analysis table
                cursor.execute('DELETE FROM analysis WHERE task_id = ?', (task_id,))

                # Delete from downloads table
                cursor.execute('DELETE FROM downloads WHERE task_id = ?', (task_id,))

                # Delete from text_index table
                cursor.execute('DELETE FROM text_index WHERE task_id = ?', (task_id,))

                # Delete from clustering table
                cursor.execute('DELETE FROM clustering WHERE task_id = ?', (task_id,))

                # Delete from cluster_details table
                cursor.execute('DELETE FROM cluster_details WHERE task_id = ?', (task_id,))

                # Commit transaction
                conn.commit()
                return True

            except sqlite3.Error as e:
                # Rollback in case of error
                conn.rollback()
                logger.error(f"Failed to delete task data: {e}")
                return False

        except sqlite3.Error as e:
            logger.error(f"Database error while deleting task data: {e}")
            return False
    def close(self):
        """Close all thread-local connections"""
        self.thread_safe_db.close()


if __name__ == '__main__':
    db = SearchEngineDatabase()
    # print(db.get_all_web_scraping())
    print(db.get_web_scraping_status(task_id='57b3fb27-1363-417f-9046-0d370d9e5169'))
