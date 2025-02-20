import os
import requests
from bs4 import BeautifulSoup

class ScrapeWeb:
    def get_html_text(self, url):
        """Scrapes the webpage and extracts useful text-based content."""
        try:
            html_data = requests.get(url, verify=False, timeout=10)
            parsed_data = BeautifulSoup(html_data.content, 'html.parser')

            # Extracting complete HTML text
            full_text = parsed_data.get_text(separator="\n", strip=True)

            # Extracting metadata
            metadata = [
                f"Title: {parsed_data.title.string}" if parsed_data.title else "Title: N/A",
                f"Meta Description: {parsed_data.find('meta', attrs={'name': 'description'})['content']}"
                if parsed_data.find("meta", attrs={"name": "description"}) else "Meta Description: N/A",
                f"Meta Keywords: {parsed_data.find('meta', attrs={'name': 'keywords'})['content']}"
                if parsed_data.find("meta", attrs={"name": "keywords"}) else "Meta Keywords: N/A"
            ]

            # Extracting all headings (H1-H6)
            headings = []
            for i in range(1, 7):
                heading_texts = [h.get_text(strip=True) for h in parsed_data.find_all(f"h{i}")]
                if heading_texts:
                    headings.append(f"\nH{i} Headings:\n" + "\n".join(heading_texts))

            # Extracting all links
            links = ["\nLinks:"] + [a['href'] for a in parsed_data.find_all('a', href=True)]

            # Extracting table data
            tables = []
            for table in parsed_data.find_all("table"):
                table_data = []
                for row in table.find_all("tr"):
                    row_data = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
                    table_data.append(" | ".join(row_data))
                if table_data:
                    tables.append("\nTable Data:\n" + "\n".join(table_data))

            # Extracting image URLs
            images = ["\nImage URLs:"] + [img['src'] for img in parsed_data.find_all('img', src=True)]

            # Combining all extracted content
            corpus_content = "\n\n".join(metadata + headings + [full_text] + links + tables + images)

            return remove_extra_empty_lines(corpus_content)

        except Exception as e:
            raise Exception(f"Error while scraping URL: {url}. Exception: {str(e)}")


def remove_extra_empty_lines(text):
    """Removes excessive empty lines for better readability."""
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    return os.linesep.join(non_empty_lines)


if __name__ == "__main__":
    try:
        # Create data storage path
        path = os.path.join(os.getcwd(), "data")
        os.makedirs(path, exist_ok=True)

        # URL to scrape (modify as needed)
        url = "https://www.khoury.northeastern.edu/masters-programs-graduate-certificates/"

        # Instantiate Scraper
        scraper = ScrapeWeb()
        extracted_text = scraper.get_html_text(url)

        # Save extracted data as a text file
        file_path = os.path.join(path, "corpus.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(extracted_text)

        print(f"✅ Data successfully scraped and stored in {file_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
