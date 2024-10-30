import json


class ScraperStore:
    def __init__(self, filename='scrapers.json'):
        self.filename = filename
        self.active_scrapers = {}

    def add_scraper(self, task_id, scraper):
        self.active_scrapers[task_id] = scraper
        self.save()

    def save(self):
        scraper_data = {}
        for task_id, scraper in self.active_scrapers.items():
            scraper_data[task_id] = scraper.__dict__

        print("sss tores")

        with open(self.filename, 'w') as f:
            json.dump(scraper_data, f)

    def load(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def get(self, task_id):
        """Get scraper data by task_id"""
        loaded_data = self.load()
        return loaded_data.get(task_id)  # Returns None if task_id not found
