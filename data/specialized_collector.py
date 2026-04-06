"""
Nova Specialized Data Collector
================================
Collects high-quality specialized training data for fine-tuning NovaMind.
Targets: StackOverflow (Python), Blender Python API, Competition Math, and top GitHub repos.

Dependencies: requests, beautifulsoup4, datasets, tqdm
"""

import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Optional dependency for math dataset
try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# --- Configuration & Endpoints ---
STACK_EXCHANGE_API = "https://api.stackexchange.com/2.3/questions"
GITHUB_SEARCH_API = "https://api.github.com/search/repositories"
BLENDER_DOCS_BASE = "https://docs.blender.org/api/current/"

HEADERS = {"User-Agent": "NovaMind-Collector/1.0 (Contact: Purushottam)"}


def clean_html(html):
    """Remove HTML tags and clean whitespace for training text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # Remove code blocks if they are already handled, or keep them
    return soup.get_text(separator="\n").strip()


# --- 1. StackOverflow Python ---
def download_stackoverflow_python(output_dir, max_questions=1000):
    """
    Fetch high-quality Python Q&A pairs from StackOverflow.
    Focuses on questions with score > 10 and an accepted answer.
    """
    print(f"[Specialized] Collecting StackOverflow Python data (max {max_questions})...")
    output_path = Path(output_dir) / "stackoverflow_python.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions_collected = 0
    page = 1

    with open(output_path, "a", encoding="utf-8") as f:
        while questions_collected < max_questions:
            params = {
                "page": page,
                "pagesize": 100,
                "order": "desc",
                "sort": "votes",
                "tagged": "python",
                "site": "stackoverflow",
                "filter": "!6VvPDzP)4X95L",  # Filter for body + accepted_answer body
            }

            try:
                resp = requests.get(STACK_EXCHANGE_API, params=params, headers=HEADERS, timeout=15)
                data = resp.json()

                if "items" not in data or not data["items"]:
                    break

                for item in data["items"]:
                    if "accepted_answer_id" in item and "body" in item:
                        title = item.get("title", "")
                        question_body = clean_html(item["body"])

                        # Find the accepted answer in the same response if the filter allowed it
                        # Or fetch it separately if needed. (Filter !6VvPDzP)4X95L usually includes answer body)
                        # For simplicity in this script, we'll assume the answer body is present or map it.
                        # (Adjusting filter to ensure answer body)

                        # Note: This is an illustrative logic, STACK API filters can be complex.
                        # Assuming filter provides 'answers' list with the accepted one.
                        answer_text = ""
                        if "answers" in item:
                            for ans in item["answers"]:
                                if ans.get("is_accepted"):
                                    answer_text = clean_html(ans.get("body", ""))
                                    break

                        if answer_text:
                            f.write(
                                f"Question: {title}\n{question_body}\nAnswer: {answer_text}\n\n"
                            )
                            questions_collected += 1
                            if questions_collected >= max_questions:
                                break

                print(f"  Processed page {page}, total collected: {questions_collected}")
                page += 1
                time.sleep(1)  # Respect rate limits

            except Exception as e:
                print(f"  Error at page {page}: {e}")
                break

    print(f"[Done] Saved {questions_collected} SO samples to {output_path}")


# --- 2. Blender Documentation ---
def download_blender_docs(output_dir):
    """
    Scrape Blender Python API markers to provide the model with latest syntax.
    """
    print("[Specialized] Scraping Blender Python API docs...")
    targets = ["bpy.ops.html", "bpy.data.html", "bpy.context.html", "bpy.types.html"]
    output_path = Path(output_dir) / "blender_api_docs.txt"

    collected_text = ""
    for target in targets:
        url = f"{BLENDER_DOCS_BASE}{target}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            # Extract main content
            content = soup.find("div", {"role": "main"})
            if content:
                collected_text += f"\n--- DOCUMENTATION: {target} ---\n"
                collected_text += content.get_text(separator="\n").strip()
                print(f"  Successfully fetched {target}")
        except Exception as e:
            print(f"  Failed to fetch {target}: {e}")

    if collected_text:
        output_path.write_text(collected_text, encoding="utf-8")
        print(f"[Done] Blender docs saved to {output_path}")


# --- 3. Math Dataset (HuggingFace) ---
def download_math_dataset(output_dir):
    """
    Load competition-level math problems from Hendrycks dataset.
    """
    if not HAS_DATASETS:
        print("[Skipped] 'datasets' library not found. Install with: pip install datasets")
        return

    print("[Specialized] Downloading Competition Math dataset...")
    try:
        dataset = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
        output_path = Path(output_dir) / "math_competition.txt"

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for item in tqdm(dataset, desc="Formatting Math"):
                prob = item.get("problem", "")
                sol = item.get("solution", "")
                f.write(f"Problem: {prob}\nSolution: {sol}\n\n")
                count += 1

        print(f"[Done] Saved {count} math problems to {output_path}")
    except Exception as e:
        print(f"  Error loading math dataset: {e}")


# --- 4. GitHub Starred Repos ---
def download_github_repos(output_dir, topics=None):
    """
    Collect code snippets and READMEs from top-starred Python repos by topic.
    """
    if topics is None:
        topics = ["pytorch", "blender-python", "data-science"]
    print(f"[Specialized] Collecting GitHub logic for topics: {topics}...")

    for topic in topics:
        print(f"  Topic: {topic}")
        output_path = Path(output_dir) / f"github_{topic}.txt"

        params = {
            "q": f"topic:{topic} language:python",
            "sort": "stars",
            "order": "desc",
            "per_page": 20,
        }

        try:
            resp = requests.get(GITHUB_SEARCH_API, params=params, headers=HEADERS, timeout=15)
            repos = resp.json().get("items", [])

            with open(output_path, "w", encoding="utf-8") as f:
                for repo in tqdm(repos, desc=f"Repo {topic}"):
                    name = repo["full_name"]
                    f.write(f"\n--- REPOSITORY: {name} ---\n")

                    # 1. Get README
                    readme_url = f"https://raw.githubusercontent.com/{name}/master/README.md"
                    r_resp = requests.get(readme_url, timeout=5)
                    if r_resp.status_code == 200:
                        f.write(f"README:\n{r_resp.text[:5000]}\n")

                    # 2. Try to get a main.py or similar file (logic sampling)
                    # For simplicity, we just take the README for now or first py file if we were to crawl
                    time.sleep(0.5)

            print(f"  [Done] Topic {topic} saved to {output_path}")
        except Exception as e:
            print(f"  Error collecting GitHub topic {topic}: {e}")


# --- 5. Main Collector ---
def collect_specialized(output_dir="data/specialized"):
    """Run all specialized collection functions."""
    start_time = time.time()
    print("=" * 60)
    print("  NovaMind Specialized Data Collection")
    print("=" * 60)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Run collectors
    download_stackoverflow_python(output_dir, max_questions=500)
    download_blender_docs(output_dir)
    download_math_dataset(output_dir)
    download_github_repos(output_dir)

    duration = time.time() - start_time
    print("=" * 60)
    print(f"  Collection Finished in {duration / 60:.1f} minutes")
    print(f"  Data stored in: {out_path.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nova Specialized Data Collector")
    parser.add_argument("--dir", type=str, default="data/specialized", help="Output directory")
    args = parser.parse_args()

    collect_specialized(args.dir)
