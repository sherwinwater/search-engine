import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup
import time


def get_article_content(url):
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Get the page content
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the article content (you might need to adjust these selectors based on the website structure)
        # This is for Yahoo Finance articles
        article_body = soup.find('div', {'class': 'caas-body'})

        if article_body:
            # Get all paragraphs
            paragraphs = article_body.find_all('p')
            content = '\n\n'.join([p.get_text().strip() for p in paragraphs])
            return content
        else:
            return "Could not extract article content"

    except Exception as e:
        return f"Error downloading article: {str(e)}"


def get_smci_news():
    try:
        smci = yf.Ticker("SMCI")
        news = smci.news

        news_data = []
        for article in news:
            article_data = {
                'date': datetime.fromtimestamp(article['providerPublishTime']),
                'title': article['title'],
                'publisher': article['publisher'],
                'link': article['link'],
                'summary': article.get('summary', 'No summary available')
            }

            # Add article content
            print(f"Downloading content for: {article['title']}")
            article_data['content'] = get_article_content(article['link'])

            # Add delay to avoid too many requests
            time.sleep(2)

            news_data.append(article_data)

        return news_data
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return None


def save_news_to_files(news_data):
    try:
        if not news_data:
            return False

        if not os.path.exists('news_data'):
            os.makedirs('news_data')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save as CSV
        df = pd.DataFrame(news_data)
        csv_filename = f'news_data/smci_news_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)

        # Save as readable TXT
        txt_filename = f'news_data/smci_news_{timestamp}.txt'
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"Super Micro Computer (SMCI) News\n")
            f.write(f"Downloaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for article in news_data:
                f.write(f"Date: {article['date']}\n")
                f.write(f"Title: {article['title']}\n")
                f.write(f"Publisher: {article['publisher']}\n")
                f.write(f"Link: {article['link']}\n")
                f.write(f"Summary: {article['summary']}\n")
                f.write("\nArticle Content:\n")
                f.write("-" * 40 + "\n")
                f.write(article['content'])
                f.write("\n" + "=" * 80 + "\n\n")

        return csv_filename, txt_filename
    except Exception as e:
        print(f"Error saving files: {str(e)}")
        return None


def main():
    try:
        print("Fetching SMCI news...")
        news_data = get_smci_news()

        if news_data:
            print("Saving news to files...")
            result = save_news_to_files(news_data)

            if result:
                csv_file, txt_file = result
                print(f"\nFiles saved successfully:")
                print(f"CSV file: {csv_file}")
                print(f"Text file: {txt_file}")

                print(f"\nCollected {len(news_data)} news articles")
                print("\nMost recent articles:")
                df = pd.DataFrame(news_data)
                print(df[['date', 'title']].head().to_string())
            else:
                print("Failed to save files.")
        else:
            print("No news data retrieved.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()