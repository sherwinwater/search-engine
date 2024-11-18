import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import time
import json
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from db.database import SearchEngineDatabase
from utils.setup_logging import setup_logging

class HTMLScraper:
    def __init__(self, base_url, output_dir, task_id=None, max_pages=51):
        parsed_url = urlparse(base_url)
        self.domain = f"{parsed_url.scheme}://{parsed_url.netloc}" if max_pages != 1 else base_url
        self.base_path = base_url
        self.output_dir = output_dir
        self.visited_urls = set()
        self.session = requests.Session()
        self.page_sizes = []
        self.request_times = []
        self.downloaded_files = []
        self.downloaded_urls = set()
        self.max_pages = max_pages
        self.db = SearchEngineDatabase()
        self.file_paths = {}  # Add this to track file paths for URLs

        self.webpage_graph_file = os.path.join(self.output_dir, 'webpage_graph.json')

        # Graph related structures
        self.graph_data = {
            'nodes': [],
            'links': []
        }
        self.url_to_id = {}

        self.logger = setup_logging(name=f"{__name__}", task_id=task_id)

        current_time = time.time()

        self.task_id = task_id
        self.status = {
            'status': 'initiated',
            'message': 'Scraper initialized',
            'webpage_graph_file': self.webpage_graph_file,
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
            },
            'graph': {
                'total_nodes': 0,
                'total_edges': 0,
                'graph_file': None
            }
        }

        # Store initial status in database
        if self.task_id:
            self.db.update_scraping_task(self.task_id, self)

    def get_url_id(self, url):
        """Generate a unique ID for a URL."""
        if url in self.url_to_id:
            return self.url_to_id[url]
        new_id = str(len(self.url_to_id) + 1)
        self.url_to_id[url] = new_id
        return new_id

    def get_absolute_path(self, url):
        """Generate absolute path for a URL and extract base filename.

        Args:
            url (str): The URL to convert to a path

        Returns:
            tuple: (absolute_path, base_filename)

        Example:
            For URL: 'https://example.com/docs/guide'
            With output_dir: '/home/user/website/content'
            Returns: ('/home/user/website/content/docs/guide/index.html', 'index.html')
        """
        # First get the path relative to domain
        relative_path = url.replace(self.domain, '').lstrip('/')

        # Add index.html if no extension
        if not os.path.splitext(relative_path)[1]:
            relative_path = os.path.join(relative_path, 'index.html')
            base_filename = 'index.html'
        else:
            base_filename = os.path.basename(relative_path)

        # Convert to absolute path using output_dir as base
        absolute_path = os.path.abspath(os.path.join(self.output_dir, relative_path))

        # Ensure the absolute path is within output_dir for security
        if not absolute_path.startswith(os.path.abspath(self.output_dir)):
            raise ValueError(f"Path {absolute_path} is outside docs directory {self.output_dir}")

        # Normalize path separators for cross-platform compatibility
        absolute_path = absolute_path.replace('\\', '/')

        return absolute_path, base_filename

    def extract_page_title(self, soup):
        """Extract page title from BeautifulSoup object."""
        if soup.title:
            return soup.title.string
        h1 = soup.find('h1')
        if h1:
            return h1.get_text()
        return None

    def save_graph_data(self):
        """Save the graph data to a JSON file."""
        graph_file = os.path.join(self.output_dir, 'webpage_graph.json')
        try:
            os.makedirs(os.path.dirname(graph_file), exist_ok=True)
            with open(graph_file, 'w', encoding='utf-8') as f:
                json.dump(self.graph_data, f, indent=2)
            self.logger.info(f"Graph data saved to {graph_file}")
            return graph_file
        except Exception as e:
            self.logger.error(f"Error saving graph data: {str(e)}")
            return None

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
        if not url or not url.startswith(self.domain):
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
                           '.js', '.ico', '.mp3', '.mp4', '.xml',
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
        # Remove fragments and whitespace
        url = url.split('#')[0].strip()

        # Skip empty URLs
        if not url:
            return None

        # Skip invalid URL types
        if url.startswith(('mailto:', 'tel:', 'javascript:', 'data:')):
            return None

        # Handle absolute URLs
        if url.startswith(('http://', 'https://')):
            return url

        # Handle root-relative URLs
        if url.startswith('/'):
            return urljoin(self.domain, url)

        # Handle relative URLs
        return urljoin(current_url, url)

    def extract_links(self, soup, current_url):
        """Extract all valid HTML links from content."""
        links = set()
        for tag in soup.find_all(['a'], href=True):
            url = tag.get('href', '').strip()
            if url:
                try:
                    absolute_url = self.normalize_url(url, current_url)
                    if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                        links.add(absolute_url)
                        self.logger.debug(f"Found valid link: {absolute_url}")
                except Exception as e:
                    self.logger.debug(f"Error processing link {url}: {str(e)}")

        self.logger.info(f"Found {len(links)} valid links on page {current_url}")
        return links

    # def analyze_page(self, url):
    #     """Analyze a single page and collect graph data."""
    #     self.logger.info(f"Starting analysis of page: {url}")
    #
    #     if url in self.visited_urls:
    #         self.logger.debug(f"Skipping already visited URL: {url}")
    #         return set()
    #
    #     self.visited_urls.add(url)
    #     url_id = self.get_url_id(url)
    #     absolute_path, file_name = self.get_absolute_path(url)  # Calculate relative path during analysis
    #
    #     try:
    #         start_time = time.time()
    #         self.logger.debug(f"Sending HEAD request to: {url}")
    #         head_response = self.session.head(url, timeout=5)
    #         content_type = head_response.headers.get('Content-Type', '').lower()
    #
    #         if 'text/html' not in content_type:
    #             self.logger.debug(f"Skipping non-HTML content type: {content_type} for {url}")
    #             return set()
    #
    #         self.logger.debug(f"Sending GET request to: {url}")
    #         response = self.session.get(url, timeout=10)
    #         response.raise_for_status()
    #
    #         request_time = time.time() - start_time
    #         page_size = len(response.content) / 1024
    #
    #         self.logger.info(f"Successfully analyzed {url} - Size: {page_size:.2f}KB, Time: {request_time:.2f}s")
    #
    #         self.request_times.append(request_time)
    #         self.page_sizes.append(page_size)
    #
    #         soup = BeautifulSoup(response.text, 'html.parser')
    #         title = self.extract_page_title(soup) or url
    #         labels = url.split('/')
    #         label = next((label for label in reversed(labels) if label), 'undefined')
    #
    #         # Calculate page weight and metadata
    #         page_weight = self.calculate_content_weight(soup, url)
    #         node_metadata = {
    #             'content_length': len(soup.get_text()),
    #             'headers_count': len(soup.find_all(['h1', 'h2', 'h3'])),
    #             'code_blocks': len(soup.find_all(['code', 'pre'])),
    #             'outbound_links': 0,
    #             'http_status': response.status_code,
    #             'content_type': content_type,
    #             'last_updated': time.time()
    #         }
    #
    #         node = {
    #             'id': url_id,
    #             'url': url,
    #             'title': title,
    #             'filename': file_name,
    #             'path': absolute_path,
    #             'label': label,
    #             'size': page_size,
    #             'weight': page_weight,
    #             'initial_rank': 1.0,
    #             'final_rank': None,
    #             'metadata': node_metadata
    #         }
    #
    #         # Store file path mapping
    #         self.file_paths[url] = absolute_path
    #
    #         if node not in self.graph_data['nodes']:
    #             self.graph_data['nodes'].append(node)
    #
    #         # Update analysis metrics
    #         self.update_status(
    #             analysis={
    #                 'pages_found': len(self.visited_urls),
    #                 'average_page_size': round(mean(self.page_sizes), 2),
    #                 'total_size_kb': round(sum(self.page_sizes), 2)
    #             }
    #         )
    #
    #         links = self.extract_links(soup, response.url)
    #         # Add edges to graph data with weights
    #         for link in links:
    #             if link not in self.url_to_id:
    #                 link_id = self.get_url_id(link)
    #             else:
    #                 link_id = self.url_to_id[link]
    #
    #             # Find the link element to calculate weight
    #             link_element = soup.find('a', href=lambda x: x and (link in x or link.endswith(x.lstrip('/'))))
    #             if link_element:
    #                 link_weight = self.calculate_link_weight(url, link, link_element)
    #             else:
    #                 link_weight = 1.0  # Default weight if link element not found
    #
    #             edge = {
    #                 'source': url_id,
    #                 'target': link_id,
    #                 'weight': link_weight,
    #                 'metadata': {
    #                     'link_text': link_element.get_text().strip() if link_element else '',
    #                     'link_class': link_element.get('class', []) if link_element else [],
    #                     'link_position': link_element.parent.name if link_element and link_element.parent else None
    #                 }
    #             }
    #
    #             if edge not in self.graph_data['links']:
    #                 self.graph_data['links'].append(edge)
    #                 node_metadata['outbound_links'] += 1  # Update outbound links count
    #
    #         self.logger.info(f"Found {len(links)} new links on page {url}")
    #         return links
    #
    #     except requests.exceptions.HTTPError as e:
    #         self.logger.error(f"HTTP Error analyzing {url}: {str(e)}")
    #         self.visited_urls.remove(url)
    #         if url in self.url_to_id:
    #             del self.url_to_id[url]
    #         return set()
    #     except Exception as e:
    #         self.logger.error(f"Error analyzing {url}: {str(e)}")
    #         self.visited_urls.remove(url)
    #         if url in self.url_to_id:
    #             del self.url_to_id[url]
    #         return set()

    def analyze_site(self, max_workers=5):
        """
        Analyze the site structure and create graph data.

        Args:
            max_workers (int): Maximum number of concurrent workers
        """
        self.logger.info(f"Starting site analysis with {max_workers} workers and {self.max_pages} page limit")
        self.status['analysis']['start_time'] = time.time()
        self.update_status('analyzing', 'Analyzing site structure...')

        try:
            # Special handling for max_pages=1: only analyze base_url
            if self.max_pages == 1:
                self.logger.info("Max pages set to 1, analyzing only base URL")
                self.analyze_page(self.base_path)

                self.save_graph_data()
                self.status['graph'].update({
                    'total_nodes': len(self.graph_data['nodes']),
                    'total_edges': len(self.graph_data['links']),
                    'graph_file': self.webpage_graph_file
                })

                self.status['analysis']['completion_time'] = time.time()
                analysis_time = self.status['analysis']['completion_time'] - self.status['analysis']['start_time']
                self.logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
                self.logger.info(f"Total pages found: 1 (max: 1)")

                self.calculate_analysis_results(analysis_time)
                return self.visited_urls

            # Original logic for max_pages > 1
            urls_to_process = {self.base_path}
            self.logger.info(f"Initial URL to process: {self.base_path}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                while urls_to_process:
                    # Rest of the original code remains the same...
                    if len(self.visited_urls) >= self.max_pages:
                        self.logger.info(f"Reached maximum page limit of {self.max_pages}. Stopping analysis.")
                        break

                    remaining_pages = self.max_pages - len(self.visited_urls)
                    batch_size = min(max_workers, remaining_pages, len(urls_to_process))
                    batch = list(urls_to_process)[:batch_size]
                    urls_to_process = set(list(urls_to_process)[batch_size:])

                    self.logger.info(f"Processing batch of {len(batch)} URLs")
                    self.logger.debug(f"Current batch: {batch}")

                    futures = [executor.submit(self.analyze_page, url) for url in batch]

                    for future in futures:
                        try:
                            new_urls = future.result()
                            if new_urls is not None and len(self.visited_urls) < self.max_pages:
                                new_unvisited_urls = {url for url in new_urls if url not in self.visited_urls}
                                urls_to_process.update(new_unvisited_urls)
                                self.logger.debug(f"Added {len(new_unvisited_urls)} new URLs to processing queue")
                        except Exception as e:
                            self.logger.error(f"Error processing batch: {str(e)}")

                    self.logger.info(f"Analysis progress: {len(self.visited_urls)}/{self.max_pages} pages found, {len(urls_to_process)} in queue")
                    self.update_status(
                        message=f"Analysis progress: {len(self.visited_urls)}/{self.max_pages} pages found",
                        analysis={
                            'pages_found': len(self.visited_urls),
                            'urls_in_queue': len(urls_to_process),
                            'pages_limit': self.max_pages
                        }
                    )

                self.save_graph_data()
                self.status['graph'].update({
                    'total_nodes': len(self.graph_data['nodes']),
                    'total_edges': len(self.graph_data['links']),
                    'graph_file': self.webpage_graph_file
                })

                self.status['analysis']['completion_time'] = time.time()
                analysis_time = self.status['analysis']['completion_time'] - self.status['analysis']['start_time']
                self.logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
                self.logger.info(f"Total pages found: {len(self.visited_urls)} (max: {self.max_pages})")

                self.calculate_analysis_results(analysis_time)
                return self.visited_urls

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            self.update_status('failed', f"Analysis failed: {str(e)}")
            return []

    def deduplicate_graph_data(self):
        """Deduplicate nodes and edges in graph data."""
        # Use dictionaries to ensure unique nodes by ID
        unique_nodes = {}
        for node in self.graph_data['nodes']:
            unique_nodes[node['id']] = node

        # Use a set with frozenset to ensure unique edges
        seen_edges = set()
        unique_edges = []
        for edge in self.graph_data['links']:
            # Create an immutable set of source and target for comparison
            edge_pair = frozenset([edge['source'], edge['target']])
            if edge_pair not in seen_edges:
                seen_edges.add(edge_pair)
                unique_edges.append(edge)

        return {
            'nodes': list(unique_nodes.values()),
            'links': unique_edges,
            'metadata': self.graph_data.get('metadata', {})
        }
    def get_graph_data(self):
        """Return the current graph data."""
        return self.graph_data

    def calculate_analysis_results(self, analysis_time):
        """Calculate and store analysis results with proper error handling."""
        self.logger.info("Starting analysis results calculation")
        total_pages = len(self.visited_urls)

        if not total_pages:
            self.logger.error("Analysis failed: No HTML pages found to analyze!")
            self.update_status('failed', 'No HTML pages found to analyze!')
            return

        try:
            # Calculate basic metrics with safety checks
            avg_page_size = mean(self.page_sizes) if self.page_sizes else 0
            total_size_kb = sum(self.page_sizes) if self.page_sizes else 0
            avg_request_time = mean(self.request_times) if self.request_times else 0
            estimated_total_time = avg_request_time * total_pages if avg_request_time > 0 else 0

            # Log basic metrics
            self.logger.info(f"Analysis Metrics:")
            self.logger.info(f"- Total Pages: {total_pages}")
            self.logger.info(f"- Average Page Size: {round(avg_page_size, 2)} KB")
            self.logger.info(f"- Total Size: {round(total_size_kb, 2)} KB ({round(total_size_kb / 1024, 2)} MB)")
            self.logger.info(f"- Average Request Time: {round(avg_request_time, 2)} seconds")
            self.logger.info(f"- Analysis Time: {round(analysis_time, 2)} seconds")

            # Check if we have any successful downloads before calculating estimates
            if avg_request_time > 0:
                self.logger.info("Estimated Completion Times:")
                self.logger.info(f"- Sequential: {round(estimated_total_time / 60, 2)} minutes")
                self.logger.info(f"- With 5 workers: {round(estimated_total_time / 5 / 60, 2)} minutes")
                self.logger.info(f"- With 10 workers: {round(estimated_total_time / 10 / 60, 2)} minutes")
            else:
                self.logger.warning("No successful requests to estimate completion times")

            # Log URL statistics
            self.logger.debug(f"Discovered URLs: {sorted(list(self.visited_urls))}")

            self.update_status(
                status='analysis_completed',
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
                    } if avg_request_time > 0 else None,
                    'discovered_urls': sorted(list(self.visited_urls))
                }
            )
            self.logger.info("Analysis results calculation completed")

        except Exception as e:
            self.logger.error(f"Error calculating analysis results: {str(e)}")
            self.update_status(
                status='analysis_error',
                message=f'Error calculating analysis results: {str(e)}',
                analysis={
                    'total_pages': total_pages,
                    'error': str(e)
                }
            )
    def download_page(self, url):
        """Download and save a single HTML page."""
        try:
            if url not in self.status['download']['current_batch']:
                self.status['download']['current_batch'].append(url)

            if url in self.downloaded_urls:
                self.status['download']['skipped_downloads'] += 1
                return True

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            absolute_path, _ = self.get_absolute_path(url)
            file_path = os.path.join(self.output_dir, absolute_path)

            if os.path.exists(file_path):
                self.status['download']['skipped_downloads'] += 1
                self.downloaded_urls.add(url)
                return True

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(response.content)

            file_info = {
                'url': url,
                'filename': absolute_path,
                'size': len(response.text),
                'timestamp': time.time()
            }

            self.downloaded_files.append(file_info)
            self.downloaded_urls.add(url)

            # Update the corresponding node in graph_data with download information
            for node in self.graph_data['nodes']:
                if node['url'] == url:
                    node.update({
                        'downloaded': True,
                        'download_timestamp': file_info['timestamp'],
                        'actual_size': file_info['size']
                    })
                    break

            # Update download progress
            self.status['download']['downloaded_pages'] += 1
            self.status['download']['successful_downloads'] += 1

            total_pages = self.status['download']['total_pages']
            total_processed = (
                self.status['download']['successful_downloads'] +
                self.status['download']['failed_downloads'] +
                self.status['download']['skipped_downloads']
            )
            self.status['download']['progress_percentage'] = (
                total_processed / total_pages * 100 if total_pages > 0 else 0
            )

            self.logger.info(f"Successfully downloaded: {url}")
            return True

        except Exception as e:
            self.status['download']['failed_downloads'] += 1
            self.status['download']['failed_urls'].append({
                'url': url,
                'error': str(e),
                'timestamp': time.time()
            })
            self.logger.error(f"Error downloading {url}: {str(e)}")
            return False
        finally:
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
            self.logger.error(f"Download process failed: {str(e)}")
            self.update_status('failed', f"Download process failed: {str(e)}")
            return {"error": str(e)}, 500

    def save_graph_data(self):
        """Save the graph data to a JSON file."""
        self.graph_data = self.deduplicate_graph_data()
        self.calculate_pagerank()

        try:
            # Create the graph data structure
            graph_data = {
                'nodes': self.graph_data['nodes'],
                'links': self.graph_data['links'],
                'metadata': {
                    'domain': self.domain,
                    'base_path': self.base_path,
                    'total_nodes': len(self.graph_data['nodes']),
                    'total_links': len(self.graph_data['links']),
                    'timestamp': time.time(),
                    'analysis_time': self.status['analysis'].get('analysis_time_seconds'),
                    'success_rate': self.status['download'].get('success_rate', 0)
                }
            }

            # Save with pretty printing
            with open(self.webpage_graph_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Graph data saved successfully to {self.webpage_graph_file }")

            # Update status
            self.status['graph'].update({
                'total_nodes': len(self.graph_data['nodes']),
                'total_edges': len(self.graph_data['links']),
                'graph_file': self.webpage_graph_file
            })

        except Exception as e:
            self.logger.error(f"Error saving graph data: {str(e)}")
            return None

    def load_graph_data(self, file_path=None):
        """Load graph data from JSON file."""
        try:
            if file_path is None:
                file_path = os.path.join(self.output_dir, 'webpage_graph.json')

            if not os.path.exists(file_path):
                self.logger.error(f"Graph file not found: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Update graph data
            self.graph_data['nodes'] = data['nodes']
            self.graph_data['links'] = data['links']

            # Rebuild URL to ID mapping
            self.url_to_id = {node['url']: node['id'] for node in data['nodes']}

            # Update status
            self.status['graph'].update({
                'total_nodes': len(self.graph_data['nodes']),
                'total_edges': len(self.graph_data['links']),
                'graph_file': file_path
            })

            self.logger.info(f"Graph data loaded successfully from {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading graph data: {str(e)}")
            return False

    def export_graph_summary(self):
        """Export a summary of the graph data."""
        try:
            summary = {
                'statistics': {
                    'total_pages': len(self.graph_data['nodes']),
                    'total_links': len(self.graph_data['links']),
                    'average_outgoing_links': len(self.graph_data['links']) / len(self.graph_data['nodes']) if self.graph_data['nodes'] else 0
                },
                'top_pages': {
                    'most_linked': self._get_most_linked_pages(5),
                    'most_outgoing': self._get_pages_with_most_outgoing_links(5)
                },
                'domain_info': {
                    'domain': self.domain,
                    'base_path': self.base_path
                }
            }

            summary_file = os.path.join(self.output_dir, 'graph_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Graph summary exported to {summary_file}")
            return summary_file

        except Exception as e:
            self.logger.error(f"Error exporting graph summary: {str(e)}")
            return None

    def _get_most_linked_pages(self, limit=5):
        """Get pages with most incoming links."""
        incoming_links = {}
        for link in self.graph_data['links']:
            target = link['target']
            incoming_links[target] = incoming_links.get(target, 0) + 1

        # Get top pages
        top_pages = sorted(incoming_links.items(), key=lambda x: x[1], reverse=True)[:limit]

        # Get page details
        return [{
            'id': page_id,
            'url': next((node['url'] for node in self.graph_data['nodes'] if node['id'] == page_id), None),
            'title': next((node['title'] for node in self.graph_data['nodes'] if node['id'] == page_id), None),
            'incoming_links': count
        } for page_id, count in top_pages]

    def _get_pages_with_most_outgoing_links(self, limit=5):
        """Get pages with most outgoing links."""
        outgoing_links = {}
        for link in self.graph_data['links']:
            source = link['source']
            outgoing_links[source] = outgoing_links.get(source, 0) + 1

        # Get top pages
        top_pages = sorted(outgoing_links.items(), key=lambda x: x[1], reverse=True)[:limit]

        # Get page details
        return [{
            'id': page_id,
            'url': next((node['url'] for node in self.graph_data['nodes'] if node['id'] == page_id), None),
            'title': next((node['title'] for node in self.graph_data['nodes'] if node['id'] == page_id), None),
            'outgoing_links': count
        } for page_id, count in top_pages]

    def calculate_content_weight(self, soup, url: str) -> float:
        """Calculate weight based on content quality and relevance."""
        weight = 1.0  # Base weight

        # 1. Title relevance
        title = soup.title.string if soup.title else ""
        if title:
            # Higher weight for pages with meaningful titles
            weight *= 1.2
            # Extra weight for main/index pages
            if any(keyword in title.lower() for keyword in ['home', 'index', 'main', 'documentation']):
                weight *= 1.5

        # 2. Content length and structure
        content_length = len(soup.get_text())
        if content_length > 5000:  # Long, detailed pages
            weight *= 1.3
        elif content_length < 500:  # Very short pages
            weight *= 0.7

        # 3. Header structure
        headers = soup.find_all(['h1', 'h2', 'h3'])
        if headers:
            weight *= (1 + min(len(headers) * 0.1, 0.5))  # Up to 50% boost for well-structured content

        # 4. URL structure
        if url.endswith(('index.html', '/', 'main.html')):
            weight *= 1.4  # Higher weight for index pages
        depth = len([x for x in url.split('/') if x]) - 2  # -2 for domain
        weight *= max(0.5, 1 - (depth * 0.1))  # Reduce weight for deeply nested pages

        # 5. Code examples and technical content
        code_blocks = len(soup.find_all(['code', 'pre']))
        if code_blocks > 0:
            weight *= (1 + min(code_blocks * 0.05, 0.3))  # Up to 30% boost for technical content

        return round(weight, 3)

    def calculate_link_weight(self, source_url: str, target_url: str, link_element) -> float:
        """Calculate weight for links between pages."""
        weight = 1.0

        # 1. Link position and visibility
        if link_element.parent and link_element.parent.name in ['nav', 'header', 'footer']:
            weight *= 1.3  # Navigation links are important

        # 2. Link text relevance
        link_text = link_element.get_text().lower()
        if any(keyword in link_text for keyword in ['documentation', 'guide', 'tutorial', 'reference']):
            weight *= 1.4

        # 3. Link attributes
        if link_element.get('class'):
            classes = ' '.join(link_element.get('class')).lower()
            if any(term in classes for term in ['nav', 'menu', 'primary']):
                weight *= 1.2

        # 4. Relative path depth difference
        source_depth = len(source_url.split('/'))
        target_depth = len(target_url.split('/'))
        depth_diff = abs(target_depth - source_depth)
        weight *= max(0.7, 1 - (depth_diff * 0.1))  # Reduce weight for big depth differences

        return round(weight, 3)

    def analyze_page(self, url):
        """Enhanced analyze_page method with weight calculations."""
        if url in self.visited_urls:
            return set()

        self.visited_urls.add(url)
        url_id = self.get_url_id(url)
        absolute_path, file_name = self.get_absolute_path(url)

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            title = self.extract_page_title(soup) or url

            # Calculate page weight
            page_weight = self.calculate_content_weight(soup, url)

            # Enhanced node data with weights
            node = {
                'id': url_id,
                'url': url,
                'title': title,
                'filename': file_name,
                'path': absolute_path,
                'label': os.path.basename(url.rstrip('/')),
                'size': len(response.content) / 1024,
                'weight': page_weight,  # Add page weight
                'initial_rank': 1.0,  # Initial PageRank score
                'final_rank': None,  # Will be calculated later
                'metadata': {
                    'content_length': len(soup.get_text()),
                    'headers_count': len(soup.find_all(['h1', 'h2', 'h3'])),
                    'code_blocks': len(soup.find_all(['code', 'pre'])),
                    'outbound_links': 0  # Will be updated
                }
            }

            if node not in self.graph_data['nodes']:
                self.graph_data['nodes'].append(node)

            # Process links with weights
            links = set()
            for link_element in soup.find_all('a', href=True):
                target_url = self.normalize_url(link_element['href'], url)

                if self.is_valid_url(target_url):
                    links.add(target_url)
                    target_id = self.get_url_id(target_url)

                    # Calculate link weight
                    link_weight = self.calculate_link_weight(url, target_url, link_element)

                    # Enhanced edge data with weights
                    edge = {
                        'source': url_id,
                        'target': target_id,
                        'weight': link_weight,
                        'metadata': {
                            'link_text': link_element.get_text().strip(),
                            'link_class': link_element.get('class', []),
                            'link_position': link_element.parent.name if link_element.parent else None
                        }
                    }

                    if edge not in self.graph_data['links']:
                        self.graph_data['links'].append(edge)

                    # Update outbound links count
                    for node in self.graph_data['nodes']:
                        if node['id'] == url_id:
                            node['metadata']['outbound_links'] += 1
                            break

            return links

        except Exception as e:
            self.logger.error(f"Error analyzing {url}: {str(e)}")
            self.visited_urls.remove(url)
            if url in self.url_to_id:
                del self.url_to_id[url]
            return set()

    def calculate_pagerank(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        """Calculate PageRank scores for all nodes."""
        try:
            n = len(self.graph_data['nodes'])
            if n == 0:
                return

            # Initialize scores
            scores = {node['id']: 1 / n for node in self.graph_data['nodes']}

            for _ in range(max_iterations):
                prev_scores = scores.copy()

                for node in self.graph_data['nodes']:
                    # Get incoming edges
                    incoming = [link for link in self.graph_data['links'] if link['target'] == node['id']]

                    # Calculate new score
                    rank = (1 - damping_factor) / n

                    for edge in incoming:
                        source_outlinks = sum(1 for link in self.graph_data['links'] if link['source'] == edge['source'])
                        if source_outlinks > 0:
                            rank += damping_factor * prev_scores[edge['source']] * edge['weight'] / source_outlinks

                    scores[node['id']] = rank

                # Check convergence
                diff = sum(abs(scores[node_id] - prev_scores[node_id]) for node_id in scores)
                if diff < tolerance:
                    break

            # Update nodes with final PageRank scores
            for node in self.graph_data['nodes']:
                node['final_rank'] = scores[node['id']]

            self.logger.info("PageRank calculation completed successfully")

        except Exception as e:
            self.logger.error(f"Error calculating PageRank: {str(e)}")

if __name__ == "__main__":
    # base_url = "https://courses.grainger.illinois.edu/cs425/fa2024/assignments.html"
    # base_url = "https://storm.apache.org/releases/2.7.0/index.html"
    # base_url = "https://visjs.github.io/vis-network/docs/network/index.html"
    base_url = "https://d3js.org/d3-force"
    output_dir = "../test/scraped_data"

    logger = setup_logging(name=f"{__name__}", task_id="test")

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
        # Print graph statistics
        graph_data = scraper.get_graph_data()
        print("\nGraph Analysis Results:")
        print(f"Total Pages (Nodes): {len(graph_data['nodes'])}")
        print(f"Total Links (Edges): {len(graph_data['links'])}")

        # Print sample of nodes
        print("\nSample Nodes:")
        for node in graph_data['nodes'][:5]:
            print(f"- {node['title']} ({node['url']})")

        print("\nSample Links:")
        for link in graph_data['links'][:5]:
            print(f"- {link['source']} -> {link['target']}")

        # Log URLs found
        logger.info("URLs found:")
        for url in scraper.visited_urls:
            logger.info(f" - {url}")

        # # Start download
        # logger.info("Starting download process...")
        # result, status_code = scraper.download_all(max_workers=10)
        #
        # # Log download results
        # if status_code == 200:
        #     logger.info("Download completed successfully:")
        #     logger.info(f" - Total URLs: {result['total_urls']}")
        #     logger.info(f" - Successfully downloaded: {result['successful_downloads']}")
        #     logger.info(f" - Skipped: {result['skipped_downloads']}")
        #     logger.info(f" - Failed: {result['failed_downloads']}")
        #
        #     if result['failed_downloads'] > 0:
        #         logger.error("Failed URLs:")
        #         for failed in result['failed_urls']:
        #             logger.error(f" - {failed['url']}: {failed['error']}")
        # else:
        #     logger.error(f"Download process failed: {result.get('error', 'Unknown error')}")
    else:
        logger.error("No pages found to analyze!")
