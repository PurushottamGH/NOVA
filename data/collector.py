"""
NovaMind Data Collector
========================
Downloads free training text automatically from multiple sources:
1. Project Gutenberg — classic public domain books
2. Wikipedia — simple English articles via API
3. arXiv — abstracts from cs.AI and astro-ph categories

All text is saved to ./personal_data/ as separate .txt files.
Handles network errors with retries and prints progress.

Usage:
    python -m data.collector
"""

import os
import time
import requests
from pathlib import Path

# Default output directory
OUTPUT_DIR = Path("personal_data")


def download_with_retry(url, max_retries=3, timeout=30):
    """
    Download content from a URL with automatic retries.
    
    Args:
        url: URL to download
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        Response text if successful, None otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"  [Retry {attempt + 1}/{max_retries}] Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None


def download_gutenberg_books(output_dir):
    """
    Download classic books from Project Gutenberg.
    
    Includes at least 5 well-known public domain texts.
    """
    # Book ID → Title mapping (Project Gutenberg plain text URLs)
    books = {
        1342: "Pride and Prejudice - Jane Austen",
        1661: "The Adventures of Sherlock Holmes - Arthur Conan Doyle",
        11: "Alice's Adventures in Wonderland - Lewis Carroll",
        84: "Frankenstein - Mary Shelley",
        1232: "The Prince - Niccolo Machiavelli",
        2701: "Moby Dick - Herman Melville",
        98: "A Tale of Two Cities - Charles Dickens",
        1952: "The Yellow Wallpaper - Charlotte Perkins Gilman",
    }

    total_chars = 0
    downloaded = 0

    print("\n=== Downloading Project Gutenberg Books ===")
    for book_id, title in books.items():
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

        print(f"  Downloading: {title}...")
        text = download_with_retry(url)
        if text is None:
            text = download_with_retry(alt_url)

        if text is not None:
            # Clean: remove Gutenberg header/footer
            start_markers = ["*** START OF", "***START OF"]
            end_markers = ["*** END OF", "***END OF"]

            for marker in start_markers:
                idx = text.find(marker)
                if idx != -1:
                    newline_idx = text.find('\n', idx)
                    if newline_idx != -1:
                        text = text[newline_idx + 1:]
                    break

            for marker in end_markers:
                idx = text.find(marker)
                if idx != -1:
                    text = text[:idx]
                    break

            filename = f"gutenberg_{book_id}_{title.split(' - ')[0].replace(' ', '_').lower()}.txt"
            filepath = output_dir / filename
            filepath.write_text(text.strip(), encoding="utf-8")

            total_chars += len(text)
            downloaded += 1
            print(f"    ✓ Saved ({len(text):,} chars)")
        else:
            print(f"    ✗ Failed to download")

    print(f"  Gutenberg total: {downloaded} books, {total_chars:,} characters")
    return total_chars


def download_wikipedia_articles(output_dir, num_articles=100):
    """
    Download simple English Wikipedia articles via the MediaWiki API.
    
    Uses the 'random' API endpoint to get diverse articles.
    """
    print(f"\n=== Downloading Wikipedia Articles ({num_articles}) ===")

    base_url = "https://en.wikipedia.org/w/api.php"
    total_chars = 0
    downloaded = 0
    batch_size = 20  # API returns up to 20 random articles at a time

    for batch in range(0, num_articles, batch_size):
        # Get random article titles
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnlimit": min(batch_size, num_articles - batch),
            "rnnamespace": 0,  # Main namespace only
        }

        try:
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("query", {}).get("random", [])
        except Exception as e:
            print(f"  Error getting article list: {e}")
            continue

        for article in articles:
            title = article["title"]
            # Get article text content
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
                "exlimit": 1,
            }

            try:
                resp = requests.get(base_url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})

                for page_id, page_data in pages.items():
                    text = page_data.get("extract", "")
                    if len(text) > 200:  # Skip very short articles
                        safe_title = "".join(c if c.isalnum() or c == '_' else '_' for c in title)[:50]
                        filename = f"wiki_{downloaded:03d}_{safe_title}.txt"
                        filepath = output_dir / filename
                        filepath.write_text(text, encoding="utf-8")
                        total_chars += len(text)
                        downloaded += 1

                        if downloaded % 20 == 0:
                            print(f"  Downloaded {downloaded}/{num_articles} articles ({total_chars:,} chars)")
            except Exception as e:
                continue

            if downloaded >= num_articles:
                break

            time.sleep(0.1)  # Be polite to the API

    print(f"  Wikipedia total: {downloaded} articles, {total_chars:,} characters")
    return total_chars


def download_arxiv_abstracts(output_dir, max_results=200):
    """
    Download paper abstracts from arXiv via their API.
    Categories: cs.AI (Artificial Intelligence) and astro-ph (Astrophysics)
    """
    print(f"\n=== Downloading arXiv Abstracts ===")

    categories = ["cs.AI", "astro-ph"]
    total_chars = 0

    for category in categories:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results // len(categories),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            content = resp.text

            # Parse XML to extract titles and abstracts
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            entries = root.findall("atom:entry", ns)
            texts = []
            for entry in entries:
                title = entry.find("atom:title", ns)
                abstract = entry.find("atom:summary", ns)
                if title is not None and abstract is not None:
                    title_text = title.text.strip().replace('\n', ' ')
                    abstract_text = abstract.text.strip().replace('\n', ' ')
                    texts.append(f"Title: {title_text}\nAbstract: {abstract_text}\n")

            if texts:
                combined = "\n".join(texts)
                safe_cat = category.replace(".", "_")
                filename = f"arxiv_{safe_cat}_abstracts.txt"
                filepath = output_dir / filename
                filepath.write_text(combined, encoding="utf-8")
                total_chars += len(combined)
                print(f"  {category}: {len(texts)} abstracts ({len(combined):,} chars)")

        except Exception as e:
            print(f"  Error downloading {category}: {e}")

        time.sleep(1)  # Be polite to arXiv

    print(f"  arXiv total: {total_chars:,} characters")
    return total_chars


def collect_all(output_dir=None):
    """
    Run all data collection pipelines.
    
    Args:
        output_dir: Directory to save text files (default: ./personal_data/)
    
    Returns:
        Total characters collected
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════╗")
    print("║    NovaMind Data Collection Pipeline     ║")
    print("╚══════════════════════════════════════════╝")

    total = 0
    total += download_gutenberg_books(output_dir)
    total += download_wikipedia_articles(output_dir, num_articles=100)
    total += download_arxiv_abstracts(output_dir, max_results=200)

    # FIXED: Count files and estimate tokens
    file_count = len(list(output_dir.glob('*.txt')))
    estimated_tokens = total // 4  # FIXED: rough estimate — ~4 chars per token for English

    print(f"\n{'='*50}")
    print(f"  TOTAL COLLECTED: {total:,} characters ({total/1e6:.1f}M)")
    print(f"  Files saved: {file_count}")  # FIXED: show file count
    print(f"  Estimated training tokens: {estimated_tokens:,}")  # FIXED: show token estimate
    print(f"  Directory: {output_dir.resolve()}")
    print(f"{'='*50}")

    return total


if __name__ == "__main__":
    collect_all()
