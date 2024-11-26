import os
import shutil

from db.database import SearchEngineDatabase
from utils.setup_logging import setup_logging

class CleanData:
    def __init__(self,task_id):
        self.db = SearchEngineDatabase()
        self.logger = setup_logging(name=f"{__name__}", task_id=task_id)
        self.task_id = task_id

    def cleanup_scraping_data(self) -> bool:
        self.logger.info(f"Starting cleanup of scraping data for task {self.task_id}")
        try:
            cursor = self.db.cursor
            cursor.execute('DELETE FROM tasks WHERE task_id = ?', (self.task_id,))
            cursor.execute('DELETE FROM analysis WHERE task_id = ?', (self.task_id,))
            cursor.execute('DELETE FROM downloads WHERE task_id = ?', (self.task_id,))

            current_dir = os.path.dirname(os.path.abspath(__file__))
            scraping_path = os.path.join(current_dir, '..', 'scraped_data', self.task_id)
            if os.path.exists(scraping_path):
                shutil.rmtree(scraping_path)
                self.logger.info(f"Removed scraping directory: {scraping_path}")

            self.db.conn.commit()
            self.logger.info(f"Successfully cleaned up scraping data for task {self.task_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error cleaning up scraping data for task {self.task_id}: {str(e)}")
            self.db.conn.rollback()
            return False

    def cleanup_index_data(self) -> bool:
        self.logger.info(f"Starting cleanup of index data for task {self.task_id}")
        try:
            index_data = self.db.get_building_text_index_status(self.task_id)

            cursor = self.db.cursor
            cursor.execute('DELETE FROM text_index WHERE task_id = ?', (self.task_id,))

            if index_data:
                if index_data['index_path'] and os.path.exists(index_data['index_path']):
                    os.remove(index_data['index_path'])
                    self.logger.info(f"Removed index file: {index_data['index_path']}")

            self.db.conn.commit()
            self.logger.info(f"Successfully cleaned up index data for task {self.task_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error cleaning up index data for task {self.task_id}: {str(e)}")
            self.db.conn.rollback()
            return False

    def cleanup_clustering_data(self) -> bool:
        self.logger.info(f"Starting cleanup of clustering data for task {self.task_id}")
        try:
            cursor = self.db.cursor
            cursor.execute('DELETE FROM clustering WHERE task_id = ?', (self.task_id,))
            cursor.execute('DELETE FROM cluster_details WHERE task_id = ?', (self.task_id,))

            self.db.conn.commit()
            self.logger.info(f"Successfully cleaned up clustering data for task {self.task_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error cleaning up clustering data for task {self.task_id}: {str(e)}")
            self.db.conn.rollback()
            return False

    def cleanup_all_data(self) -> bool:
        self.logger.info(f"===Starting cleanup of all data for task {self.task_id}===")
        success = True

        try:
            self.db.conn.execute('BEGIN TRANSACTION')

            clustering_success = self.cleanup_clustering_data()
            index_success = self.cleanup_index_data()
            scraping_success = self.cleanup_scraping_data()

            success = scraping_success and index_success and clustering_success

            if success:
                self.db.conn.commit()
                self.logger.info(f"Successfully cleaned up all data for task {self.task_id}")
            else:
                self.db.conn.rollback()
                self.logger.warning(f"Cleanup failed for task {self.task_id}, rolling back changes")

        except Exception as e:
            self.logger.error(f"Error in cleanup_all_data for task {self.task_id}: {str(e)}")
            self.db.conn.rollback()
            success = False

        finally:
            self.db.close()

        return success