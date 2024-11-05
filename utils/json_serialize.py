def prepare_scraper_status_for_json(status_data):
    """Prepare scraper status data for JSON serialization with selected fields only"""
    if not status_data:
        return None

    # Create a new dictionary with only the fields we want
    filtered_status = {
        'task_id': status_data.get('task_id'),
        'status': status_data.get('status'),
        'message': status_data.get('message'),
        'data_location': status_data.get('output_dir'),
        'webpage_graph_file': status_data.get('webpage_graph_file'),
        'is_completed': status_data.get('is_completed'),
        'downloaded_urls': list(status_data.get('downloaded_urls', set())),
        # 'downloaded_files': status_data.get('downloaded_files') or [],

        # Include selected nested fields from analysis
        'analysis': {
            'pages_found': status_data.get('analysis', {}).get('pages_found', 0),
            'total_size_kb': status_data.get('analysis', {}).get('total_size_kb', 0),
            'completion_time': status_data.get('analysis', {}).get('completion_time')
        },

        # Include selected nested fields from download
        'download': {
            'total_pages': status_data.get('download', {}).get('total_pages', 0),
            'downloaded_pages': status_data.get('download', {}).get('downloaded_pages', 0),
            'successful_downloads': status_data.get('download', {}).get('successful_downloads', 0),
            'failed_downloads': status_data.get('download', {}).get('failed_downloads', 0),
            'skipped_downloads': status_data.get('download', {}).get('skipped_downloads', 0),
            'progress_percentage': status_data.get('download', {}).get('progress_percentage', 0),
            'is_completed': status_data.get('download', {}).get('is_completed', False)
        }
    }

    return filtered_status

def prepare_index_status_for_json(status_data):
    """Prepare index status data for JSON serialization"""
    if not status_data:
        return None

    return {
        'task_id': status_data['task_id'],
        'status': status_data['status'],
        'message': status_data['message'],
        'index_info': {
            'docs_dir': status_data['docs_dir'],
            'index_path': status_data['index_path']
        },
        'progress': {
            'total_files': status_data.get('total_files', 0),
            'processed_files': status_data.get('processed_files', 0),
            'failed_files': status_data.get('failed_files', 0),
            'progress_percentage': round(status_data.get('progress_percentage', 0), 2),
            'current_file': status_data.get('current_file')
        },
        'timing': {
            'start_time': status_data.get('start_time'),
            'completion_time': status_data.get('completion_time'),
            'created_at': status_data.get('created_at'),
            'updated_at': status_data.get('updated_at')
        },
        'is_completed': bool(status_data.get('is_completed', False)),
        'error': status_data.get('error')
    }
