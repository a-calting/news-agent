"""
Personal News Agent.

What it does, in plain terms:
  1. Fetches the latest headlines from several reputable RSS feeds.
  2. Sends all the headlines to an AI model (Google Gemini, free tier).
  3. The AI sorts each important story into one of your categories, decides
     which stories genuinely matter, and writes a one-sentence French summary.
  4. Prints the most important stories, grouped by category.

Setup (one time):
  - Put your free Gemini API key in the .env file (get one at
    https://aistudio.google.com/apikey).
  - Install the libraries:  python3 -m pip install -r requirements.txt

Run it with:  python3 news.py
"""

import os                           # reads the API key from your environment
import json                         # reads the AI's structured (JSON) reply
import urllib.request               # downloads the RSS feeds from the web
import xml.etree.ElementTree as ET  # reads the RSS (XML) format

from dotenv import load_dotenv      # loads your key from the .env file
from google import genai            # the official Google Gemini library
from google.genai import types      # helper types for configuring the request

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# How many headlines to pull from each source.
MAX_HEADLINES = 10

# Which Gemini model to use. gemini-2.5-flash is free and fast.
# (You could switch this to "gemini-3.5-flash" later — also free.)
GEMINI_MODEL = "gemini-2.5-flash"

# Your categories. The AI must place each story into exactly one of these.
CATEGORIES = [
    "environment", "economy", "AI", "politics", "animals",
    "stocks", "finance", "tech", "military", "health", "other",
]

# The news sources. Each has a friendly name and an RSS feed address.
SOURCES = [
    {
        "name": "Reuters (via Google News)",
        "url": "https://news.google.com/rss/search?q=world+news+site:reuters.com&hl=en-US&gl=US&ceid=US:en",
    },
    {
        "name": "BBC World",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
    },
    {
        "name": "The Guardian World",
        "url": "https://www.theguardian.com/world/rss",
    },
    {
        "name": "NPR News",
        "url": "https://feeds.npr.org/1001/rss.xml",
    },
    {
        "name": "CNBC Finance",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    },
    {
        "name": "BBC Science & Environment",
        "url": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    },
    {
        "name": "Ars Technica (Technology)",
        "url": "https://feeds.arstechnica.com/arstechnica/index",
    },
    {
        "name": "BBC Health",
        "url": "https://feeds.bbci.co.uk/news/health/rss.xml",
    },
]

# What we tell the AI to do. This is the "job description" for the brain.
SYSTEM_INSTRUCTION = f"""\
You are a news editor. You will be given a numbered list of news headlines from
several sources. Do the following:

1. Read every headline.
2. Decide which stories are genuinely important (significant, consequential,
   or widely impactful). Ignore minor, repetitive, or trivial items. Be
   selective — return only the stories that truly matter, not all of them.
3. Sort each important story into exactly ONE of these categories:
   {", ".join(CATEGORIES)}.
   Use "other" only if none of the others fit.
4. For each important story, write a ONE-SENTENCE summary in FRENCH.

Refer to each story by the integer id it was given. Do not invent stories.
"""


def fetch_headlines(url):
    """Download one feed and return a list of (headline, link) pairs."""
    # Identify ourselves as a normal browser; some feeds reject other clients.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        raw_data = response.read()

    root = ET.fromstring(raw_data)

    headlines = []
    # In an RSS feed, every article is an <item> with a <title> and <link>.
    for item in root.iter("item"):
        title = item.findtext("title")
        link = item.findtext("link")
        if title and link:
            headlines.append((title, link))
        if len(headlines) >= MAX_HEADLINES:
            break
    return headlines


def collect_all_articles():
    """Fetch every source and return one combined, numbered list of articles.

    Each article is a dict: {"id", "source", "title", "link"}.
    The id is what we send to the AI so it can refer to stories without us
    having to send (or trust it to repeat) the long, messy links.
    """
    articles = []
    next_id = 1
    for source in SOURCES:
        try:
            headlines = fetch_headlines(source["url"])
        except Exception as error:
            print(f"  (Could not load {source['name']}: {error})")
            continue
        for title, link in headlines:
            articles.append({
                "id": next_id,
                "source": source["name"],
                "title": title,
                "link": link,
            })
            next_id += 1
    return articles


def analyze_with_gemini(articles, api_key):
    """Send the articles to Gemini and get back the important ones, sorted.

    Returns a list of dicts: {"id", "category", "summary_fr"}.
    """
    # Build the numbered list of headlines to hand to the AI.
    headline_lines = [
        f"[{a['id']}] ({a['source']}) {a['title']}" for a in articles
    ]
    prompt = "Here are today's headlines:\n\n" + "\n".join(headline_lines)

    # Describe the exact shape of reply we want, so we get clean data back
    # instead of free-form text. This is called a "response schema".
    response_schema = {
        "type": "object",
        "properties": {
            "stories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "category": {"type": "string", "enum": CATEGORIES},
                        "summary_fr": {"type": "string"},
                    },
                    "required": ["id", "category", "summary_fr"],
                },
            },
        },
        "required": ["stories"],
    }

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )

    data = json.loads(response.text)
    return data.get("stories", [])


def print_grouped(stories, articles):
    """Print the selected stories grouped by category."""
    # Let us look up an article's title/link quickly by its id.
    by_id = {a["id"]: a for a in articles}

    # Group the chosen stories by category.
    by_category = {}
    for story in stories:
        article = by_id.get(story.get("id"))
        if article is None:
            continue  # skip any id the AI returned that we don't recognize
        by_category.setdefault(story["category"], []).append((story, article))

    # Print categories in the order you listed them, skipping empty ones.
    for category in CATEGORIES:
        items = by_category.get(category)
        if not items:
            continue
        print()
        print("=" * 70)
        print(category.upper())
        print("=" * 70)
        for story, article in items:
            print(f"\n• {article['title']}")
            print(f"  FR: {story['summary_fr']}")
            print(f"  {article['link']}")
    print()


def main():
    # Load the API key from the .env file into the environment.
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        print("No Gemini API key found.")
        print("Get a free key at https://aistudio.google.com/apikey, then paste")
        print("it into the .env file (replace PASTE_YOUR_KEY_HERE).")
        return

    print("Fetching headlines...")
    articles = collect_all_articles()
    if not articles:
        print("No headlines could be fetched.")
        return

    print(f"Sending {len(articles)} headlines to Gemini for sorting...")
    try:
        stories = analyze_with_gemini(articles, api_key)
    except Exception as error:
        print(f"The AI request failed: {error}")
        return

    if not stories:
        print("The AI did not return any stories.")
        return

    print_grouped(stories, articles)


if __name__ == "__main__":
    main()
