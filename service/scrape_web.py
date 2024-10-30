import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from db.database import SearchEngineDatabase
from utils.setup_logging import setup_logging

logger = setup_logging(__name__)

class HTMLScraper:
    def __init__(self, base_url, output_dir, task_id=None):
        parsed_url = urlparse(base_url)
        self.domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.base_path = os.path.dirname(base_url)
        self.output_dir = output_dir
        self.visited_urls = set()
        self.session = requests.Session()
        self.page_sizes = []
        self.request_times = []
        self.downloaded_files = []
        self.downloaded_urls = set()
        self.db = SearchEngineDatabase()

        current_time = time.time()

        self.task_id = task_id
        self.status = {
            'status': 'initiated',
            'message': 'Scraper initialized',
            'start_time': current_time,
            'is_completed': False,
            'completion_time': None,
            'analysis': {
                'pages_found': 0,
                'urls_in_queue': 0,
                'average_page_size': 0,
                'total_size_kb': 0,
                'average_request_time': 0,
                'start_time': None,
                'completion_time': None
            },
            'download': {
                'total_pages': 0,
                'downloaded_pages': 0,
                'successful_downloads': 0,
                'failed_downloads': 0,
                'skipped_downloads': 0,
                'current_batch': [],
                'failed_urls': [],
                'progress_percentage': 0,
                'start_time': None,
                'completion_time': None,
                'is_completed': False
            }
        }

        # Store initial status in database
        if self.task_id:
            self.db.update_scraping_task(self.task_id, self)

    def update_status(self, status=None, message=None, **kwargs):
        """Update scraper status and any additional fields."""
        if status:
            self.status['status'] = status
        if message:
            self.status['message'] = message

        for key, value in kwargs.items():
            if '.' in key:
                main_key, sub_key = key.split('.')
                if main_key in self.status:
                    self.status[main_key][sub_key] = value
            else:
                if isinstance(value, dict) and key in self.status and isinstance(self.status[key], dict):
                    self.status[key].update(value)
                else:
                    self.status[key] = value

        # Update database after status change
        if self.task_id:
            self.db.update_scraping_task(self.task_id, self)

    def get_status(self):
        """Get current status with latest metrics."""
        self.status['analysis'].update({
            'pages_found': len(self.visited_urls),
            'average_page_size': round(mean(self.page_sizes), 2) if self.page_sizes else 0,
            'total_size_kb': round(sum(self.page_sizes), 2) if self.page_sizes else 0,
            'average_request_time': round(mean(self.request_times), 2) if self.request_times else 0
        })

        # Get latest status from database if available
        if self.task_id:
            db_status = self.db.get_web_scraping_status(self.task_id)
            if db_status:
                # Update scraper instance with database data
                self.domain = db_status['domain']
                self.base_path = db_status['base_path']
                self.output_dir = db_status['output_dir']
                self.visited_urls = db_status['visited_urls']
                self.page_sizes = db_status['page_sizes']
                self.request_times = db_status['request_times']
                self.downloaded_files = db_status['downloaded_files']
                self.downloaded_urls = db_status['downloaded_urls']
                self.status.update(db_status)

        return self.status

    def __del__(self):
        """Cleanup database connection on object destruction."""
        if hasattr(self, 'db'):
            self.db.close()
    def is_download_completed(self):
        """Check if download process is completed."""
        download_status = self.status['download']
        total_processed = (
                download_status['successful_downloads'] +
                download_status['failed_downloads'] +
                download_status['skipped_downloads']
        )
        return total_processed >= download_status['total_pages']

    def get_download_progress(self):
        """Get detailed download progress information."""
        download_status = self.status['download']
        total_pages = download_status['total_pages']

        if total_pages == 0:
            return {
                'message': 'No pages to download',
                'progress_percentage': 0,
                'is_completed': False
            }

        total_processed = (
                download_status['successful_downloads'] +
                download_status['failed_downloads'] +
                download_status['skipped_downloads']
        )

        progress_percentage = (total_processed / total_pages * 100) if total_pages > 0 else 0

        return {
            'message': f"Processed {total_processed} of {total_pages} pages",
            'progress_percentage': round(progress_percentage, 2),
            'successful': download_status['successful_downloads'],
            'failed': download_status['failed_downloads'],
            'skipped': download_status['skipped_downloads'],
            'remaining': total_pages - total_processed,
            'is_completed': self.is_download_completed(),
            'elapsed_time': time.time() - self.status['start_time']
        }

    def is_valid_url(self, url):
        """Check if URL should be processed."""
        if not url or not url.startswith(self.base_path):
            return False

        # Define excluded URL patterns
        excluded_patterns = [
            'https://storm.apache.org/talksAndVideos.html',
            # Add more patterns here as needed
        ]

        # Check if URL starts with any excluded pattern
        if any(url.startswith(pattern) for pattern in excluded_patterns):
            return False

        # Skip non-HTML resource links and anchors
        skip_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.svg', '.css',
                           '.js', '.ico', '.mp3', '.mp4', '.pdf', '.xml',
                           '.json', '.woff', '.woff2', '.ttf', '.eot'}
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False

        # Skip certain types of links
        if any(url.startswith(prefix) for prefix in ['#', 'mailto:', 'javascript:', 'tel:']):
            return False

        parsed = urlparse(url)
        if not parsed.netloc:
            return True
        return parsed.netloc in self.domain

    def normalize_url(self, url, current_url):
        """Normalize URL to absolute form."""
        url = url.split('#')[0]
        if url.startswith(('http://', 'https://')):
            return url
        if url.startswith('/'):
            return urljoin(self.domain, url)
        return urljoin(current_url, url)

    def extract_links(self, soup, current_url):
        """Extract all valid HTML links from content."""
        links = set()
        for tag in soup.find_all('a', href=True):
            url = tag.get('href')
            if url:
                absolute_url = self.normalize_url(url, current_url)
                if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                    links.add(absolute_url)
        return links

    def analyze_page(self, url):
        """Analyze a single page without downloading."""
        if url in self.visited_urls:
            return set()

        self.visited_urls.add(url)

        try:
            start_time = time.time()

            head_response = self.session.head(url, timeout=5)
            content_type = head_response.headers.get('Content-Type', '').lower()

            if 'text/html' not in content_type:
                return set()

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            request_time = time.time() - start_time
            page_size = len(response.content) / 1024  # Convert to KB

            self.request_times.append(request_time)
            self.page_sizes.append(page_size)



            soup = BeautifulSoup(response.text, 'html.parser')

            # Update analysis metrics
            self.update_status(
                analysis={
                    'pages_found': len(self.visited_urls),
                    'average_page_size': round(mean(self.page_sizes), 2),
                    'total_size_kb': round(sum(self.page_sizes), 2)
                }
            )

            return self.extract_links(soup, response.url)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # logger.info(f"Page not found: {url},removed from visited_urls")
                self.visited_urls.remove(url)
            else:
                logger.error(f"Error analyzing {url}: {str(e)}")
        except Exception as e:
            logger.error(f"Error analyzing {url}: {str(e)}")
            return set()

    def analyze_site(self, max_workers=5):
        """Analyze the site structure and estimate download requirements."""
        self.status['analysis']['start_time'] = time.time()
        self.update_status('analyzing', 'Analyzing site structure...')

        try:
            urls_to_process = {self.base_path}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                while urls_to_process:
                    batch = list(urls_to_process)[:max_workers]
                    urls_to_process = set(list(urls_to_process)[max_workers:])

                    futures = [executor.submit(self.analyze_page, url) for url in batch]

                    for future in futures:
                        try:
                            new_urls = future.result()
                            if new_urls is not None:
                                urls_to_process.update(
                                    url for url in new_urls
                                    if url not in self.visited_urls
                                )
                        except Exception as e:
                            logger.error(f"Error processing batch: {str(e)}")

                    self.update_status(
                        message=f"Analysis progress: {len(self.visited_urls)} pages found",
                        analysis={
                            'pages_found': len(self.visited_urls),
                            'urls_in_queue': len(urls_to_process)
                        }
                    )

            self.status['analysis']['completion_time'] = time.time()
            analysis_time = self.status['analysis']['completion_time'] - self.status['analysis']['start_time']
            self.calculate_analysis_results(analysis_time)
            return len(self.visited_urls) > 0

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            self.update_status('failed', f"Analysis failed: {str(e)}")
            return False


    def calculate_analysis_results(self, analysis_time):
        """Calculate and store analysis results."""
        total_pages = len(self.visited_urls)

        if not total_pages:
            self.update_status('failed', 'No HTML pages found to analyze!')
            return

        avg_page_size = mean(self.page_sizes)
        total_size_kb = sum(self.page_sizes)
        avg_request_time = mean(self.request_times)
        estimated_total_time = avg_request_time * total_pages

        self.update_status(
            message='Analysis completed',
            analysis={
                'total_pages': total_pages,
                'average_page_size_kb': round(avg_page_size, 2),
                'total_size_kb': round(total_size_kb, 2),
                'total_size_mb': round(total_size_kb / 1024, 2),
                'average_request_time': round(avg_request_time, 2),
                'analysis_time_seconds': round(analysis_time, 2),
                'estimated_times': {
                    'sequential_minutes': round(estimated_total_time / 60, 2),
                    'parallel_5_workers_minutes': round(estimated_total_time / 5 / 60, 2),
                    'parallel_10_workers_minutes': round(estimated_total_time / 10 / 60, 2)
                },
                'discovered_urls': sorted(list(self.visited_urls))
            }
        )

    def download_page(self, url):
        """Download and save a single HTML page."""
        try:
            # Update current batch information
            if url not in self.status['download']['current_batch']:
                self.status['download']['current_batch'].append(url)

            # Check if URL has already been downloaded
            if url in self.downloaded_urls:
                self.status['download']['skipped_downloads'] += 1
                # logger.info(f"Skipping already downloaded URL: {url}")
                return True

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            relative_path = url.replace(self.domain, '').lstrip('/')
            if not os.path.splitext(relative_path)[1]:
                relative_path = os.path.join(relative_path, 'index.html')

            file_path = os.path.join(self.output_dir, relative_path)

            if os.path.exists(file_path):
                self.status['download']['skipped_downloads'] += 1
                # logger.info(f"File already exists: {file_path}")
                self.downloaded_urls.add(url)
                return True

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(response.content)

            file_info = {
                'url': url,
                'filename': relative_path,
                'size': len(response.text),
                'timestamp': time.time()
            }

            self.downloaded_files.append(file_info)
            self.downloaded_urls.add(url)

            # Update download progress
            self.status['download']['downloaded_pages'] += 1
            self.status['download']['successful_downloads'] += 1

            # Update progress percentage
            total_pages = self.status['download']['total_pages']
            total_processed = (
                    self.status['download']['successful_downloads'] +
                    self.status['download']['failed_downloads'] +
                    self.status['download']['skipped_downloads']
            )
            self.status['download']['progress_percentage'] = (
                total_processed / total_pages * 100 if total_pages > 0 else 0
            )

            logger.info(f"Successfully downloaded: {url}")
            return True

        except Exception as e:
            self.status['download']['failed_downloads'] += 1
            self.status['download']['failed_urls'].append({
                'url': url,
                'error': str(e),
                'timestamp': time.time()
            })
            logger.error(f"Error downloading {url}: {str(e)}")
            return False
        finally:
            # Remove URL from current batch
            if url in self.status['download']['current_batch']:
                self.status['download']['current_batch'].remove(url)

    def download_all(self, max_workers=5):
        """Download all analyzed pages."""
        if not self.visited_urls:
            error_msg = "No URLs to download. Run analyze_site() first!"
            self.update_status('failed', error_msg)
            return {"error": error_msg}, 400

        total_urls = len(self.visited_urls)
        current_time = time.time()

        # Update download status with all required fields
        self.status['download'].update({
            'total_pages': total_urls,
            'downloaded_pages': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped_downloads': 0,
            'current_batch': [],
            'failed_urls': [],
            'progress_percentage': 0,
            'start_time': current_time,
            'is_completed': False
        })

        self.update_status(
            'downloading',
            f"Starting download of {total_urls} pages..."
        )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.download_page, url)
                           for url in self.visited_urls
                           if url not in self.downloaded_urls]

                for future in futures:
                    future.result()

            # Update completion status and timing
            completion_time = time.time()
            self.status['download']['completion_time'] = completion_time
            self.status['download']['is_completed'] = True
            self.status['completion_time'] = completion_time
            self.status['is_completed'] = True

            # Calculate final statistics
            download_stats = self.status['download']
            success_rate = (download_stats['successful_downloads'] / total_urls * 100) if total_urls > 0 else 0

            completion_message = (
                f"Download completed: {download_stats['successful_downloads']} of {total_urls} "
                f"pages downloaded successfully ({download_stats['skipped_downloads']} skipped, "
                f"{download_stats['failed_downloads']} failed)"
            )

            self.update_status('completed', completion_message)

            return {
                "total_urls": total_urls,
                "successful_downloads": download_stats['successful_downloads'],
                "skipped_downloads": download_stats['skipped_downloads'],
                "failed_downloads": download_stats['failed_downloads'],
                "failed_urls": download_stats['failed_urls'],
                "success_rate": round(success_rate, 2),
                "downloaded_files": self.downloaded_files,
                "completion_time": completion_time,
                "total_time": completion_time - self.status['download']['start_time']
            }, 200

        except Exception as e:
            logger.error(f"Download process failed: {str(e)}")
            self.update_status('failed', f"Download process failed: {str(e)}")
            return {"error": str(e)}, 500

if __name__ == "__main__":
    base_url = "https://storm.apache.org/releases/2.7.0/index.html"
    output_dir = "../test/scraped_data"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    scraper = HTMLScraper(base_url, output_dir, task_id="test")

    # Log analysis progress
    logger.info("Starting site analysis...")
    analysis_result = scraper.analyze_site(max_workers=10)
    scraper.calculate_analysis_results(scraper.status['analysis']['completion_time']-scraper.status['analysis']['start_time'])
    logger.info(f"Analysis completed. Found {len(scraper.visited_urls)} URLs")
    logger.info(f"status: {scraper.status}")

    if analysis_result:
        # Log URLs found
        logger.info("URLs found:")
        for url in scraper.visited_urls:
            logger.info(f" - {url}")

        # Start download
        logger.info("Starting download process...")
        result, status_code = scraper.download_all(max_workers=10)

        # Log download results
        if status_code == 200:
            logger.info("Download completed successfully:")
            logger.info(f" - Total URLs: {result['total_urls']}")
            logger.info(f" - Successfully downloaded: {result['successful_downloads']}")
            logger.info(f" - Skipped: {result['skipped_downloads']}")
            logger.info(f" - Failed: {result['failed_downloads']}")

            if result['failed_downloads'] > 0:
                logger.error("Failed URLs:")
                for failed in result['failed_urls']:
                    logger.error(f" - {failed['url']}: {failed['error']}")
        else:
            logger.error(f"Download process failed: {result.get('error', 'Unknown error')}")
    else:
        logger.error("No pages found to analyze!")
