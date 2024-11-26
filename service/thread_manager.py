from threading import Thread, local
from typing import Dict, Optional
import threading
import os

from db.database import SearchEngineDatabase
from service.build_text_index import BuildTextIndex
from service.clean_data import CleanData
from service.document_clustering import DocumentClustering
from service.scrape_web import HTMLScraper
from utils.setup_logging import setup_logging

class ThreadManager:
    """Manages background threads and their lifecycle"""

    def __init__(self):
        self._active_threads: Dict[str, Thread] = {}
        self._local = local()
        self.logger = setup_logging(__name__)

    def register_thread(self, task_id: str, thread: Thread) -> None:
        """Register a thread for tracking"""
        self.logger.info(f"Registering thread {task_id}")
        self._active_threads[task_id] = thread

    def get_thread(self, task_id: str) -> Optional[Thread]:
        """Get thread by task ID"""
        self.logger.info(f"Getting thread {task_id}")
        return self._active_threads.get(task_id)

    def remove_thread(self, task_id: str) -> None:
        """Remove thread from tracking"""
        if task_id in self._active_threads:
            del self._active_threads[task_id]

    def is_thread_alive(self, task_id: str) -> bool:
        """Check if a thread is still running"""
        thread = self.get_thread(task_id)
        return thread is not None and thread.is_alive()

    def start_background_scraping(self, url: str, output_dir: str, task_id: str,
                                max_workers: int = 10, max_pages: int = 50) -> None:
        """Start background scraping task"""
        def _scraping_wrapper():
            thread = threading.current_thread()
            try:
                db = SearchEngineDatabase()
                scraper = HTMLScraper(base_url=url, output_dir=output_dir,
                                    task_id=task_id, max_pages=max_pages)
                db.update_scraping_task(task_id, scraper)

                if not getattr(thread, "cancelled", False):
                    if scraper.analyze_site(max_workers=max_workers):
                        if not getattr(thread, "cancelled", False):
                            scraper.download_all(max_workers=max_workers)
            except Exception as e:
                scraper.update_status('failed', f'Error during scraping: {str(e)}')
            finally:
                db.close()
                self.remove_thread(task_id)

        thread = Thread(target=_scraping_wrapper)
        self.register_thread(task_id, thread)
        thread.start()

    def start_background_indexing(self, docs_dir: str, task_id: str, url: str) -> None:
        """Start background indexing task"""
        def _indexing_wrapper():
            thread = threading.current_thread()
            try:
                if not getattr(thread, "cancelled", False):
                    buildTextIndex = BuildTextIndex(docs_dir=docs_dir,
                                                  task_id=task_id,
                                                  scraping_url=url)
                    buildTextIndex.load_documents()

                    if not getattr(thread, "cancelled", False):
                        buildTextIndex.build_index()

                    if not getattr(thread, "cancelled", False):
                        buildTextIndex.save_index()
            except Exception as e:
               self.logger.exception(e)
            finally:
                self.remove_thread(task_id)

        thread = Thread(target=_indexing_wrapper)
        self.register_thread(task_id, thread)
        thread.start()

    def start_background_clustering(self, docs_dir: str, task_id: str, url: str) -> None:
        """Start background clustering task"""
        def _clustering_wrapper():
            thread = threading.current_thread()
            try:
                db = SearchEngineDatabase()
                try:
                    if not getattr(thread, "cancelled", False):
                        db.update_clustering_status(task_id, 'processing')
                        webpage_graph = os.path.join(docs_dir, 'webpage_graph.json')

                        cluster = DocumentClustering(task_id, docs_dir, webpage_graph)
                        cluster.build_clustering_data()

                        if not getattr(thread, "cancelled", False):
                            db.update_clustering_status(task_id, 'completed')

                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(f"Error in clustering for task {task_id}: {error_msg}")
                    db.update_clustering_status(task_id, 'failed', error_msg)
                finally:
                    db.close()
            except Exception as e:
               self.logger.error(f"Critical error in clustering thread: {str(e)}")
            finally:
                self.remove_thread(task_id)

        thread = Thread(target=_clustering_wrapper)
        self.register_thread(task_id, thread)
        thread.start()

    def process_url_pipeline(self, url: str, destination_path: str, task_id: str, max_pages: int) -> None:
        """Start the complete URL processing pipeline"""
        def _pipeline_wrapper():
            thread = threading.current_thread()
            self.register_thread(task_id, thread)
            db = SearchEngineDatabase()

            try:
                # Check cancellation before each major step
                if not getattr(thread, "cancelled", False):
                    # Scraping
                    self.start_background_scraping(url, destination_path, task_id, 10, max_pages)
                    # Wait for scraping to complete
                    scraping_thread = self.get_thread(task_id)
                    if scraping_thread:
                        scraping_thread.join()

                if not getattr(thread, "cancelled", False):
                    # Indexing
                    self.start_background_indexing(destination_path, task_id, url)
                    # Wait for indexing to complete
                    indexing_thread = self.get_thread(task_id)
                    if indexing_thread:
                        indexing_thread.join()

                if not getattr(thread, "cancelled", False):
                    # Clustering
                    self.start_background_clustering(destination_path, task_id, url)
                    # Wait for clustering to complete
                    clustering_thread = self.get_thread(task_id)
                    if clustering_thread:
                        clustering_thread.join()

            except Exception as e:
                error_msg = f"Pipeline error for {task_id}: {str(e)}"
                self.logger.error(error_msg)

            finally:
                # Clean up thread registration
                self.remove_thread(task_id)
                db.close()

        # Start the pipeline in a new thread
        pipeline_thread = Thread(target=_pipeline_wrapper)
        self.register_thread(task_id, pipeline_thread)
        pipeline_thread.start()

    def start_pipeline(self, url: str, destination_path: str, task_id: str, max_pages: int) -> None:
        """Convenience method to start the pipeline process"""
        self.process_url_pipeline(url, destination_path, task_id, max_pages)

    def start_cleanup(self, task_id: str) -> None:
        """Start background cleanup task"""
        def _cleanup_task():
            thread = threading.current_thread()
            try:
                db = SearchEngineDatabase()
                try:
                    if not getattr(thread, "cancelled", False):
                        clean_data = CleanData(task_id)
                        clean_data.cleanup_all_data()
                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(f"Error in cleanup for task {task_id}: {error_msg}")
                finally:
                    db.close()
            except Exception as e:
                self.logger.error(f"Critical error in cleanup thread: {str(e)}")
            finally:
                self.remove_thread(f"{task_id}_cleanup")

        thread = Thread(target=_cleanup_task)
        self.register_thread(f"{task_id}_cleanup", thread)
        thread.start()
    def kill_thread(self, task_id: str) -> bool:
        """Kill a running thread and clean up"""
        thread = self.get_thread(task_id)
        if thread and thread.is_alive():
            setattr(thread, "cancelled", True)
            thread.join(timeout=2.0)
            self.remove_thread(task_id)
            self.logger.info(f"Killed thread {task_id}")
            return True
        return False

    def kill_all_threads(self) -> None:
        """Kill all running threads"""
        for task_id in list(self._active_threads.keys()):
            self.kill_thread(task_id)