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

import time
from pathlib import Path

import requests

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

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
            headers = {"User-Agent": "NovaMindDataBot/1.0 (https://github.com/PurushottamGH/NOVA)"}
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"  [Retry {attempt + 1}/{max_retries}] Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
    return None


def download_gutenberg_books(output_dir):
    """
    Download classic books from Project Gutenberg.

    Includes at least 5 well-known public domain texts.
    """
    # Top 50 most popular books ID → Title mapping (Project Gutenberg)
    books = {
        "84": "Frankenstein; or, the modern prometheus by Mary Wollstonecraft Shelley",
        "45304": "The City of God, Volume I by Saint of Hippo Augustine",
        "33283": "Calculus Made Easy by Silvanus P. Thompson",
        "2701": "Moby Dick; Or, The Whale by Herman Melville",
        "1342": "Pride and Prejudice by Jane Austen",
        "52106": "The origin and development of the moral ideas by Edward Westermarck",
        "8492": "The King in Yellow by Robert W. Chambers",
        "1513": "Romeo and Juliet by William Shakespeare",
        "58031": "Report of the President's Commission on the Assassination of President John F. Kennedy",
        "64317": "The Great Gatsby by F. Scott Fitzgerald",
        "11": "Alice's Adventures in Wonderland by Lewis Carroll",
        "43": "The strange case of Dr. Jekyll and Mr. Hyde by Robert Louis Stevenson",
        "2641": "A Room with a View by E. M. Forster",
        "768": "Wuthering Heights by Emily Brontë",
        "100": "The Complete Works of William Shakespeare by William Shakespeare",
        "2554": "Crime and Punishment by Fyodor Dostoyevsky",
        "1260": "Jane Eyre: An Autobiography by Charlotte Brontë",
        "844": "The Importance of Being Earnest: A Trivial Comedy for Serious People by Oscar Wilde",
        "174": "The Picture of Dorian Gray by Oscar Wilde",
        "145": "Middlemarch by George Eliot",
        "1184": "The Count of Monte Cristo by Alexandre Dumas and Auguste Maquet",
        "26471": "Spoon River Anthology by Edgar Lee Masters",
        "67979": "The Blue Castle: a novel by L. M. Montgomery",
        "68348": "England under the Angevin Kings, Volumes I and II by Kate Norgate",
        "37106": "Little Women; Or, Meg, Jo, Beth, and Amy by Louisa May Alcott",
        "345": "Dracula by Bram Stoker",
        "98": "A Tale of Two Cities by Charles Dickens",
        "2542": "A Doll's House : a play by Henrik Ibsen",
        "28942": "The Junior Classics, Volume 1: Fairy and wonder tales by William Allan Neilson",
        "5197": "My Life — Volume 1 by Richard Wagner",
        "47715": "The Works of William Shakespeare [Cambridge Edition] [Vol. 7 of 9] by William Shakespeare",
        "13188": "Putnam's Word Book by Louis A. Flemming",
        "16389": "The Enchanted April by Elizabeth Von Arnim",
        "1259": "Twenty years after by Alexandre Dumas and Auguste Maquet",
        "1661": "The Adventures of Sherlock Holmes by Arthur Conan Doyle",
        "1080": "A Modest Proposal by Jonathan Swift",
        "6761": "The Adventures of Ferdinand Count Fathom — Complete by T. Smollett",
        "76": "Adventures of Huckleberry Finn by Mark Twain",
        "2160": "The Expedition of Humphry Clinker by T. Smollett",
        "57336": "Ancient Britain and the Invasions of Julius Caesar by T. Rice Holmes",
        "37683": "Chambers's Twentieth Century Dictionary (part 1 of 4: A-D)",
        "6593": "History of Tom Jones, a Foundling by Henry Fielding",
        "394": "Cranford by Elizabeth Cleghorn Gaskell",
        "50559": "The Works of William Shakespeare [Cambridge Edition] [Vol. 3 of 9] by William Shakespeare",
        "4085": "The Adventures of Roderick Random by T. Smollett",
        "9701": "I. Beowulf: an Anglo-Saxon poem. II. The fight at Finnsburh: a fragment.",
        "39647": "The Spanish American Reader by Ernesto Nelson",
        "205": "Walden, and On The Duty Of Civil Disobedience by Henry David Thoreau",
        "67792": "A History of Magic and Experimental Science, Volume 1 (of 2) by Lynn Thorndike",
        "49008": "The Works of William Shakespeare [Cambridge Edition] [Vol. 8 of 9] by William Shakespeare",
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
                    newline_idx = text.find("\n", idx)
                    if newline_idx != -1:
                        text = text[newline_idx + 1 :]
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
            print("    ✗ Failed to download")

    print(f"  Gutenberg total: {downloaded} books, {total_chars:,} characters")
    return total_chars


def download_wikipedia_articles(output_dir, num_articles=50000):
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
            headers = {"User-Agent": "NovaMindDataBot/1.0 (https://github.com/PurushottamGH/NOVA)"}
            resp = requests.get(base_url, params=params, timeout=15, headers=headers)
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
                headers = {
                    "User-Agent": "NovaMindDataBot/1.0 (https://github.com/PurushottamGH/NOVA)"
                }
                resp = requests.get(base_url, params=params, timeout=15, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})

                for _page_id, page_data in pages.items():
                    text = page_data.get("extract", "")
                    if len(text) > 200:  # Skip very short articles
                        safe_title = "".join(c if c.isalnum() or c == "_" else "_" for c in title)[
                            :50
                        ]
                        filename = f"wiki_{downloaded:03d}_{safe_title}.txt"
                        filepath = output_dir / filename
                        filepath.write_text(text, encoding="utf-8")
                        total_chars += len(text)
                        downloaded += 1

                        if downloaded % 20 == 0:
                            print(
                                f"  Downloaded {downloaded}/{num_articles} articles ({total_chars:,} chars)"
                            )
            except Exception:
                continue

            if downloaded >= num_articles:
                break

            time.sleep(0.1)  # Be polite to the API

    print(f"  Wikipedia total: {downloaded} articles, {total_chars:,} characters")
    return total_chars


def download_arxiv_abstracts(output_dir, max_results=10000):
    """
    Download paper abstracts from arXiv via their API.
    Categories: cs.AI (Artificial Intelligence) and astro-ph (Astrophysics)
    """
    print("\n=== Downloading arXiv Abstracts ===")

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
            headers = {"User-Agent": "NovaMindDataBot/1.0 (https://github.com/PurushottamGH/NOVA)"}
            resp = requests.get(url, params=params, timeout=30, headers=headers)
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
                    title_text = title.text.strip().replace("\n", " ")
                    abstract_text = abstract.text.strip().replace("\n", " ")
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


def download_gsm8k(output_dir):
    """Math word problems — 8.5K train examples"""
    if not HAS_DATASETS:
        print("\n[Skip] GSM8K: 'datasets' library not installed.")
        return 0
    from datasets import load_dataset

    print("\n=== Downloading GSM8K ===")
    try:
        ds = load_dataset("gsm8k", "main", split="train")

        texts = []
        for item in ds:
            texts.append(f"<|user|>\n{item['question']}\n<|assistant|>\n{item['answer']}\n")

        out = Path(output_dir) / "gsm8k_math.txt"
        out.write_text("\n".join(texts), encoding="utf-8")
        print(f"  ✓ GSM8K: {len(texts)} math problems saved")
        return len("\n".join(texts))
    except Exception as e:
        print(f"  ✗ GSM8K failed: {e}")
        return 0


def download_alpaca(output_dir):
    """52K instruction-following pairs"""
    if not HAS_DATASETS:
        print("\n[Skip] Alpaca: 'datasets' library not installed.")
        return 0
    from datasets import load_dataset

    print("\n=== Downloading Alpaca ===")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")

        texts = []
        for item in ds:
            instruction = item["instruction"]
            if item.get("input"):
                instruction += f"\n{item['input']}"
            texts.append(f"<|user|>\n{instruction}\n<|assistant|>\n{item['output']}\n")

        out = Path(output_dir) / "alpaca_sft.txt"
        out.write_text("\n".join(texts), encoding="utf-8")
        print(f"  ✓ Alpaca: {len(texts)} instruction pairs saved")
        return len("\n".join(texts))
    except Exception as e:
        print(f"  ✗ Alpaca failed: {e}")
        return 0


def download_openassistant(output_dir):
    """High quality human conversations"""
    if not HAS_DATASETS:
        print("\n[Skip] OpenAssistant: 'datasets' library not installed.")
        return 0
    from datasets import load_dataset

    print("\n=== Downloading OpenAssistant ===")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")

        # Filter to only assistant messages with high quality
        texts = []
        for item in ds:
            if item["role"] == "assistant" and item.get("rank", 1) == 0:
                texts.append(item["text"])

        out = Path(output_dir) / "openassistant_sft.txt"
        out.write_text("\n".join(texts), encoding="utf-8")
        print(f"  ✓ OpenAssistant: {len(texts)} responses saved")
        return len("\n".join(texts))
    except Exception as e:
        print(f"  ✗ OpenAssistant failed: {e}")
        return 0


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
    total += download_wikipedia_articles(output_dir, num_articles=50000)
    total += download_arxiv_abstracts(output_dir, max_results=10000)

    # 200M Config Additions: Math, Code, SFT
    total += download_gsm8k(output_dir)
    total += download_alpaca(output_dir)
    total += download_openassistant(output_dir)

    # FIXED: Count files and estimate tokens
    file_count = len(list(output_dir.glob("*.txt")))
    estimated_tokens = total // 4  # FIXED: rough estimate — ~4 chars per token for English

    print(f"\n{'=' * 50}")
    print(f"  TOTAL COLLECTED: {total:,} characters ({total / 1e6:.1f}M)")
    print(f"  Files saved: {file_count}")  # FIXED: show file count
    print(f"  Estimated training tokens: {estimated_tokens:,}")  # FIXED: show token estimate
    print(f"  Directory: {output_dir.resolve()}")
    print(f"{'=' * 50}")

    return total


if __name__ == "__main__":
    collect_all()
