import os
import sqlite3
import json
import time
from datetime import datetime
from threading import local

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
                    error TEXT
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
                    status, message, start_time, is_completed, completion_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                scraper.status['completion_time']
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
            base_path = os.path.dirname(url)

            # Get task data
            cursor.execute('SELECT * FROM tasks WHERE base_path = ?', (base_path,))
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

            # Get task data
            cursor.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
            task_data = cursor.fetchone()
            print(task_id)
            if not task_data:
                print("----none")
                return None

            # Get analysis data
            cursor.execute('SELECT * FROM analysis WHERE task_id = ?', (task_id,))
            analysis_data = cursor.fetchone()

            # Get download data
            cursor.execute('SELECT * FROM downloads WHERE task_id = ?', (task_id,))
            download_data = cursor.fetchone()

            # Reconstruct complete status
            return {
                'task_id': task_data[0],
                'domain': task_data[1],
                'base_path': task_data[2],
                'output_dir': task_data[3],
                'visited_urls': json.loads(task_data[4]),  # Return as list, not set
                'page_sizes': json.loads(task_data[5]),
                'request_times': json.loads(task_data[6]),
                'downloaded_files': json.loads(task_data[7]),
                'downloaded_urls': json.loads(task_data[8]),  # Return as list, not set
                'status': task_data[9],
                'message': task_data[10],
                'start_time': task_data[11],
                'is_completed': bool(task_data[12]),
                'completion_time': task_data[13],
                'analysis': {
                    'pages_found': analysis_data[1],
                    'urls_in_queue': analysis_data[2],
                    'average_page_size': analysis_data[3],
                    'total_size_kb': analysis_data[4],
                    'average_request_time': analysis_data[5],
                    'start_time': analysis_data[6],
                    'completion_time': analysis_data[7]
                } if analysis_data else {},
                'download': {
                    'total_pages': download_data[1],
                    'downloaded_pages': download_data[2],
                    'successful_downloads': download_data[3],
                    'failed_downloads': download_data[4],
                    'skipped_downloads': download_data[5],
                    'current_batch': json.loads(download_data[6]),
                    'failed_urls': json.loads(download_data[7]),
                    'progress_percentage': download_data[8],
                    'start_time': download_data[9],
                    'completion_time': download_data[10],
                    'is_completed': bool(download_data[11])
                } if download_data else {}
            }
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
                    is_completed, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                getattr(text_index, 'error', None)
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
                       is_completed, error
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
                    'error': result[15]
                }
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving building status: {e}")
            return None

    def get_all_building_text_index(self):
        """Get all text index building statuses"""
        try:
            self.cursor.execute('''
                SELECT *
                FROM text_index 
            ''')

            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            logger.error(f"Error retrieving building status: {e}")
            return []

    def close(self):
        """Close all thread-local connections"""
        self.thread_safe_db.close()


if __name__ == '__main__':
    db = SearchEngineDatabase()
    # print(db.get_all_web_scraping())
    print(db.get_web_scraping_status(task_id='57b3fb27-1363-417f-9046-0d370d9e5169'))
