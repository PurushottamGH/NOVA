"""
Nova Web Search
==================
Web search and content fetching for Nova.
Uses DuckDuckGo HTML search (no API key) and arXiv API for papers.

Dependencies: requests, beautifulsoup4

Usage:
    searcher = NovaWebSearch()
    results = searcher.search("transformer architecture explained")
    context = searcher.search_and_summarize("attention is all you need")
"""

import re
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from urllib.parse import quote_plus, urljoin

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


# Common headers to avoid being blocked
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# arXiv API base URL
ARXIV_API = "http://export.arxiv.org/api/query"


class NovaWebSearch:
    """
    Web search engine for Nova.

    - DuckDuckGo HTML search (no API key required)
    - Page fetching with clean text extraction
    - arXiv paper search via the public API
    """

    def __init__(self, timeout: int = 10):
        if not HAS_DEPS:
            raise ImportError(
                "NovaWebSearch requires 'requests' and 'beautifulsoup4'. "
                "Install with: pip install requests beautifulsoup4"
            )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ------------------------------------------------------------------ #
    #  DuckDuckGo search
    # ------------------------------------------------------------------ #

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search the web using DuckDuckGo HTML interface.

        Args:
            query: Search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of dicts with keys: title, url, snippet
        """
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[NovaWebSearch] Search request failed: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        # DuckDuckGo HTML results are in div.result
        for item in soup.select(".result"):
            if len(results) >= num_results:
                break

            # Title and URL
            title_tag = item.select_one(".result__a")
            snippet_tag = item.select_one(".result__snippet")

            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)

            # DuckDuckGo wraps URLs in a redirect — extract the real URL
            href = title_tag.get("href", "")
            if "uddg=" in href:
                # Extract actual URL from redirect parameter
                match = re.search(r'uddg=([^&]+)', href)
                if match:
                    from urllib.parse import unquote
                    href = unquote(match.group(1))

            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            results.append({
                "title": title,
                "url": href,
                "snippet": snippet,
            })

        return results

    # ------------------------------------------------------------------ #
    #  Page fetching
    # ------------------------------------------------------------------ #

    def fetch_page(self, url: str, max_chars: int = 3000) -> str:
        """
        Fetch a URL and extract clean text content.

        Args:
            url: URL to fetch.
            max_chars: Maximum characters to return.

        Returns:
            Clean text content from the page.
        """
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            return f"[Error fetching {url}: {e}]"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script, style, nav, footer, header elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        # Extract text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        # Truncate to max_chars
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + "\n[...truncated]"

        return clean_text

    # ------------------------------------------------------------------ #
    #  Search + summarize
    # ------------------------------------------------------------------ #

    def search_and_summarize(self, query: str) -> str:
        """
        Search the web and return combined context from top results.

        Args:
            query: Search query string.

        Returns:
            Formatted string with source titles and snippets.
        """
        results = self.search(query, num_results=3)

        if not results:
            return f"No results found for: {query}"

        parts = []
        for r in results:
            snippet = r["snippet"]
            # If snippet is too short, try fetching the page
            if len(snippet) < 50 and r["url"]:
                try:
                    page_text = self.fetch_page(r["url"], max_chars=500)
                    if page_text and not page_text.startswith("[Error"):
                        snippet = page_text
                except Exception:
                    pass

            parts.append(f"Source: {r['title']}\n{snippet}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  arXiv search
    # ------------------------------------------------------------------ #

    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search arXiv for academic papers.

        Args:
            query: Search query (e.g. "transformer attention mechanism").
            max_results: Maximum number of papers to return.

        Returns:
            List of dicts with keys: title, abstract, authors, url, date
        """
        params = {
            "search_query": f"all:{quote_plus(query)}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            resp = self.session.get(ARXIV_API, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[NovaWebSearch] arXiv request failed: {e}")
            return []

        # Parse the Atom XML feed
        root = ET.fromstring(resp.text)

        # arXiv uses Atom namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)

            # Get authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name_el = author.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # Get the abstract link (prefer abs URL)
            paper_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "text/html":
                    paper_url = link.get("href", "")
                    break
            if not paper_url:
                id_el = entry.find("atom:id", ns)
                if id_el is not None and id_el.text:
                    paper_url = id_el.text.strip()

            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            # Clean up multiline titles/abstracts
            title = re.sub(r'\s+', ' ', title)

            abstract = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
            abstract = re.sub(r'\s+', ' ', abstract)

            date = published_el.text.strip()[:10] if published_el is not None and published_el.text else ""

            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "url": paper_url,
                "date": date,
            })

        return papers
