"""
Personal News Agent.

What it does, in plain terms:
  1. Fetches the latest headlines from several reputable RSS feeds (today/yesterday).
  2. Sends all the headlines to an AI model (Google Gemini, free tier).
  3. The AI sorts every story into one of your categories (in French), decides
     which deserve a phone notification, and picks one lead story of the day.
  4. Pushes short mini/medium notifications to your phone via ntfy.
  5. Builds a full digest web page (lead story + everything grouped by category),
     publishes it to GitHub Pages, and sends a "Résumé du jour" notification
     whose tap opens that page.

Setup (one time):
  - Put your free Gemini API key in the .env file (get one at
    https://aistudio.google.com/apikey).
  - Install the libraries:  python3 -m pip install -r requirements.txt

Run it with:  python3 news.py
"""

import os                           # reads the API key from your environment
import json                         # reads the AI's structured (JSON) reply
import time                         # lets us pause briefly between notifications
import html                         # safely escapes text for the web page
import subprocess                   # runs git to publish the page to GitHub Pages
import urllib.request               # downloads the RSS feeds from the web
import xml.etree.ElementTree as ET  # reads the RSS (XML) format

from datetime import datetime, timedelta   # for "is this story recent?" checks
from zoneinfo import ZoneInfo               # so all times are in YOUR timezone, not the server's
from email.utils import parsedate_to_datetime  # reads RSS dates like "Sun, 15 Jun 2026 13:00:00 GMT"

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

# French names for the categories, used as the headings on the digest page.
CATEGORY_LABELS_FR = {
    "environment": "Environnement",
    "economy": "Économie",
    "AI": "Intelligence artificielle",
    "politics": "Politique",
    "animals": "Animaux",
    "stocks": "Bourse",
    "finance": "Finance",
    "tech": "Technologie",
    "military": "Militaire",
    "health": "Santé",
    "other": "Autres",
}

# French month names, for a nicely formatted date at the top of the page.
FRENCH_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]

# Your timezone. Every date and time in this script is computed in this zone,
# so it behaves the same whether it runs on your laptop or on GitHub's servers
# (which use UTC). It also automatically handles summer/winter time.
LOCAL_TZ = ZoneInfo("Europe/Paris")

# Quiet hours: no notifications are sent at or after QUIET_START (11pm) or
# before QUIET_END (8am), your local time. The agent still runs and publishes
# the digest page during these hours — it just stays silent.
QUIET_START = 23   # 11 pm
QUIET_END = 8      # 8 am

# Where the published digest lives, and the public web address GitHub Pages
# serves it at. The script writes the page into docs/index.html and pushes it.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(REPO_DIR, "docs")
DIGEST_PATH = os.path.join(DOCS_DIR, "index.html")
PAGE_URL = "https://a-calting.github.io/news-agent/"

# A small local file remembering which stories were already pushed, so mini and
# medium notifications never repeat across runs. Entries older than a week are
# pruned automatically. (This file stays on your computer; it is not published.)
SEEN_PATH = os.path.join(REPO_DIR, "notified.json")
SEEN_RETENTION_DAYS = 7

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
several sources. Everything you write for the reader must be in FRENCH.

STEP 1 — Sort EVERY headline.
Return one entry in "stories" for every headline in the list, with:
  - "id": the integer id it was given.
  - "category": exactly ONE of these: {", ".join(CATEGORIES)}.
       Use "other" only if none of the others fit.
  - "headline_fr": a short, clear French headline (a few words).
  - "body_fr": a short French summary — one sentence, at most two.
  - "notify": whether this story should be pushed to the phone right now:
       * "medium": a notable story worth a short push (one or two sentences).
       * "micro":  a minor but interesting story worth a one-line push.
       * "none":   not worth its own push; it still appears in the digest.
Be selective with pushes: MOST stories should be "none". Only use "micro" or
"medium" for stories that genuinely deserve an immediate notification.

STEP 2 — Choose ONE lead story.
Pick the single most important, substantial story of the day to feature at the
top of the digest. IMPORTANT: the lead MUST be a story you marked "none" — it is
reserved for the full read, not already pushed as a notification. Return it in
"lead", in FRENCH, with:
  - "id": its integer id (must match one of the stories above whose notify is "none").
  - "headline_fr": a strong French headline.
  - "writeup_fr": the FULLEST French write-up — 3 to 6 sentences giving the
       background, the key facts, and why it matters.

Do not invent stories. Refer to every story by its given integer id.
"""


def fetch_headlines(url):
    """Download one feed and return a list of (headline, link, published) items.

    `published` is the article's publication date as a datetime, or None if the
    feed didn't provide one (or it couldn't be read).
    """
    # Identify ourselves as a normal browser; some feeds reject other clients.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        raw_data = response.read()

    root = ET.fromstring(raw_data)

    headlines = []
    # In an RSS feed, every article is an <item> with a <title> and <link>,
    # and usually a <pubDate> saying when it was published.
    for item in root.iter("item"):
        title = item.findtext("title")
        link = item.findtext("link")

        published = None
        pub_text = item.findtext("pubDate")
        if pub_text:
            try:
                published = parsedate_to_datetime(pub_text)
            except (TypeError, ValueError):
                published = None

        if title and link:
            headlines.append((title, link, published))
        if len(headlines) >= MAX_HEADLINES:
            break
    return headlines


def now_local():
    """The current date and time in your timezone (Europe/Paris)."""
    return datetime.now(LOCAL_TZ)


def in_quiet_hours(moment=None):
    """True if it's quiet hours now (23:00–08:00 Europe/Paris) — stay silent."""
    moment = moment or now_local()
    return moment.hour >= QUIET_START or moment.hour < QUIET_END


def is_recent(published):
    """True if the article is from today or yesterday (keeps the news fresh).

    If we couldn't read a date, we keep the story rather than risk dropping a
    real, recent headline just because its feed left the date out.
    """
    if published is None:
        return True
    yesterday = now_local().date() - timedelta(days=1)
    return published.astimezone(LOCAL_TZ).date() >= yesterday


def format_date(published):
    """A short, readable date for display, e.g. "15/06/2026" (empty if unknown)."""
    if published is None:
        return ""
    return published.astimezone(LOCAL_TZ).strftime("%d/%m/%Y")


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
        for title, link, published in headlines:
            # Skip anything older than yesterday — we only want fresh news.
            if not is_recent(published):
                continue
            articles.append({
                "id": next_id,
                "source": source["name"],
                "title": title,
                "link": link,
                "date_str": format_date(published),
            })
            next_id += 1
    return articles


def analyze_with_gemini(articles, api_key):
    """Send the articles to Gemini and get back the sorted result.

    Returns a dict with two keys:
      - "lead":    {"id", "headline_fr", "writeup_fr"} — the single top story.
      - "stories": a list of {"id", "category", "notify", "headline_fr", "body_fr"}
                   covering every headline we sent.
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
            "lead": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "headline_fr": {"type": "string"},
                    "writeup_fr": {"type": "string"},
                },
                "required": ["id", "headline_fr", "writeup_fr"],
            },
            "stories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "category": {"type": "string", "enum": CATEGORIES},
                        "notify": {"type": "string", "enum": ["micro", "medium", "none"]},
                        "headline_fr": {"type": "string"},
                        "body_fr": {"type": "string"},
                    },
                    "required": ["id", "category", "notify", "headline_fr", "body_fr"],
                },
            },
        },
        "required": ["lead", "stories"],
    }

    client = genai.Client(api_key=api_key)

    # The free tier can briefly return "high demand" (503) errors. If that
    # happens, wait a few seconds and try again, up to a few times.
    last_error = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            return json.loads(response.text)
        except Exception as error:
            last_error = error
            if attempt < 3:
                print(f"  (Gemini busy, attempt {attempt} failed — retrying in 10s...)")
                time.sleep(10)

    # All attempts failed; let the caller report the problem.
    raise last_error


def send_ntfy_notification(topic, title, body, click=None):
    """Send one push notification to your phone via ntfy.

    We use ntfy's JSON format: we POST a small JSON object to https://ntfy.sh/
    that names the topic, the title, and the message. Anyone subscribed to that
    topic in the ntfy phone app receives it.

    Why JSON instead of HTTP headers? French text uses accents and special
    characters (é, ç, «», —). Those are safe inside a JSON body but can break
    an HTTP header, so JSON is the reliable choice here.

    If `click` is given, tapping the notification opens that web address.
    """
    message = {
        "topic": topic,      # which channel to send to
        "title": title,      # the bold heading shown on your phone
        "message": body,     # the notification text
    }
    if click:
        message["click"] = click  # web page opened when you tap the notification

    payload = json.dumps(message).encode("utf-8")
    request = urllib.request.Request(
        "https://ntfy.sh/",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(request, timeout=15)


def group_by_category(stories, articles, skip_id=None):
    """Group stories by category, in your preferred category order.

    Returns a list of (category, [(story, article), ...]) pairs, skipping empty
    categories and (optionally) the one story whose id is `skip_id` (the lead).
    """
    by_id = {a["id"]: a for a in articles}
    buckets = {}
    for story in stories:
        if skip_id is not None and story.get("id") == skip_id:
            continue
        article = by_id.get(story.get("id"))
        if article is None:
            continue  # skip any id the AI returned that we don't recognize
        buckets.setdefault(story["category"], []).append((story, article))

    return [(c, buckets[c]) for c in CATEGORIES if c in buckets]


def french_date_today():
    """Today's date in French, e.g. "15 juin 2026"."""
    now = now_local()
    return f"{now.day} {FRENCH_MONTHS[now.month - 1]} {now.year}"


def print_grouped(data, articles):
    """Print the lead story, then every story grouped by category."""
    by_id = {a["id"]: a for a in articles}
    lead = data.get("lead") or {}
    lead_id = lead.get("id")

    # The lead story, at the top.
    print()
    print("#" * 70)
    print("À LA UNE")
    print("#" * 70)
    if lead.get("headline_fr"):
        lead_article = by_id.get(lead_id)
        date_str = lead_article.get("date_str", "") if lead_article else ""
        suffix = f" ({date_str})" if date_str else ""
        print(f"\n{lead['headline_fr']}{suffix}")
        print(lead.get("writeup_fr", ""))
        if lead_article:
            print(lead_article["link"])

    # Everything else, grouped by category (the lead is left out here).
    for category, items in group_by_category(data.get("stories", []), articles, skip_id=lead_id):
        print()
        print("=" * 70)
        print(CATEGORY_LABELS_FR.get(category, category).upper())
        print("=" * 70)
        for story, article in items:
            date_str = article.get("date_str", "")
            date_suffix = f" ({date_str})" if date_str else ""
            tag = " [déjà notifié]" if story.get("notify") in ("micro", "medium") else ""
            print(f"\n• {story['headline_fr']}{date_suffix}{tag}")
            print(f"  {story['body_fr']}")
            print(f"  {article['link']}")
    print()


# The look of the digest page. Kept in a plain string so the curly braces in
# the CSS don't clash with Python's f-string formatting below.
DIGEST_CSS = """
  * { box-sizing: border-box; }
  body { margin: 0; background: #f6f7f9; color: #1a1a1a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.55; }
  header { background: #0f172a; color: #fff; padding: 28px 20px; }
  header h1 { margin: 0; font-size: 1.6rem; }
  header .today { margin: 6px 0 0; color: #cbd5e1; font-size: .95rem; }
  main { max-width: 720px; margin: 0 auto; padding: 20px 16px 60px; }
  .lead { background: #fff; border-radius: 14px; padding: 22px; margin: 18px 0 30px;
    box-shadow: 0 2px 10px rgba(0,0,0,.06); border-left: 5px solid #e11d48; }
  .lead .kicker { margin: 0 0 6px; color: #e11d48; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; font-size: .72rem; }
  .lead h2 { margin: 0 0 4px; font-size: 1.45rem; line-height: 1.25; }
  .lead p { margin: .6em 0; }
  .cat { margin: 26px 0; }
  .cat h3 { font-size: 1.05rem; text-transform: uppercase; letter-spacing: .05em;
    color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }
  .card { background: #fff; border-radius: 12px; padding: 14px 16px; margin: 12px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,.05); }
  .card h4 { margin: 0 0 6px; font-size: 1.02rem; }
  .card p { margin: .35em 0; }
  .meta { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .date { color: #64748b; font-size: .82rem; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .src { display: inline-block; margin-top: 8px; font-weight: 600; }
  .badge { display: inline-block; font-size: .66rem; font-weight: 700;
    background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 999px;
    text-transform: uppercase; letter-spacing: .04em; vertical-align: middle; }
  footer { text-align: center; color: #94a3b8; font-size: .8rem; padding: 24px; }
"""


def generate_digest_html(data, articles):
    """Build the full digest as a single HTML page (returned as text).

    The page has two parts: a featured lead story at the top, then every other
    kept story grouped by category. Stories already pushed as a mini/medium
    notification get a "déjà notifié" badge so you know you can skip them.
    """
    by_id = {a["id"]: a for a in articles}
    esc = html.escape
    lead = data.get("lead") or {}
    lead_id = lead.get("id")

    # --- The lead story ---
    lead_html = ""
    if lead.get("headline_fr"):
        lead_article = by_id.get(lead_id)
        lead_date = esc(lead_article.get("date_str", "")) if lead_article else ""
        writeup = lead.get("writeup_fr", "")
        # Turn blank-line-separated text into separate paragraphs.
        chunks = [c.strip() for c in writeup.split("\n\n") if c.strip()] or [writeup]
        paragraphs = "".join(f"<p>{esc(c)}</p>" for c in chunks)
        link_html = ""
        if lead_article:
            link_html = f'<a class="src" href="{esc(lead_article["link"])}">Lire l\'article original →</a>'
        lead_html = f"""
    <section class="lead">
      <p class="kicker">À la une</p>
      <h2>{esc(lead["headline_fr"])}</h2>
      <p class="date">{lead_date}</p>
      {paragraphs}
      {link_html}
    </section>"""

    # --- Every other story, grouped by category ---
    sections = []
    for category, items in group_by_category(data.get("stories", []), articles, skip_id=lead_id):
        cards = []
        for story, article in items:
            badge = ('<span class="badge">déjà notifié</span>'
                     if story.get("notify") in ("micro", "medium") else "")
            cards.append(f"""
      <article class="card">
        <h4>{esc(story["headline_fr"])} {badge}</h4>
        <p>{esc(story["body_fr"])}</p>
        <p class="meta"><span class="date">{esc(article.get("date_str", ""))}</span>
          <a href="{esc(article["link"])}">Lire l'article →</a></p>
      </article>""")
        label = esc(CATEGORY_LABELS_FR.get(category, category))
        sections.append(f'\n    <section class="cat">\n      <h3>{label}</h3>{"".join(cards)}\n    </section>')

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Résumé du jour</title>
<style>{DIGEST_CSS}</style>
</head>
<body>
  <header>
    <h1>Résumé du jour</h1>
    <p class="today">{esc(french_date_today())}</p>
  </header>
  <main>{lead_html}{"".join(sections)}
  </main>
  <footer>Généré automatiquement par votre agent d'actualités</footer>
</body>
</html>
"""


def publish_to_github_pages(html_text):
    """Save the page to docs/index.html and push it to GitHub so Pages serves it.

    Returns True if the page was pushed (or was already up to date), False if
    publishing failed. Either way the page is saved locally in docs/index.html.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(DIGEST_PATH, "w", encoding="utf-8") as page_file:
        page_file.write(html_text)

    def git(*git_args):
        return subprocess.run(
            ["git", *git_args], cwd=REPO_DIR, capture_output=True, text=True
        )

    git("add", "docs/index.html")

    # If the page is identical to what's already committed, there's nothing
    # to push — that's still a success.
    if git("diff", "--cached", "--quiet").returncode == 0:
        return True

    commit = git("commit", "-m", "Update daily digest")
    if commit.returncode != 0:
        print(f"  (git commit failed: {commit.stderr.strip()})")
        return False

    push = git("push")
    if push.returncode != 0:
        print(f"  (git push failed: {push.stderr.strip()})")
        return False
    return True


def load_notified():
    """Return {article link: "YYYY-MM-DD"} for stories already pushed to the phone.

    Entries older than SEEN_RETENTION_DAYS are dropped so the file stays small.
    A missing or unreadable file just means "nothing sent yet".
    """
    try:
        with open(SEEN_PATH, encoding="utf-8") as seen_file:
            stored = json.load(seen_file).get("notified", {})
    except (FileNotFoundError, ValueError):
        return {}

    cutoff = now_local().date() - timedelta(days=SEEN_RETENTION_DAYS)
    fresh = {}
    for link, day in stored.items():
        try:
            when = datetime.strptime(day, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue  # skip any malformed entry
        if when >= cutoff:
            fresh[link] = day
    return fresh


def save_notified(notified):
    """Save the {article link: date} record of stories we've pushed."""
    try:
        with open(SEEN_PATH, "w", encoding="utf-8") as seen_file:
            json.dump({"notified": notified}, seen_file, ensure_ascii=False, indent=2)
    except OSError as error:
        print(f"  (Could not save {SEEN_PATH}: {error})")


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
        data = analyze_with_gemini(articles, api_key)
    except Exception as error:
        print(f"The AI request failed: {error}")
        return

    stories = data.get("stories", [])
    if not stories:
        print("The AI did not return any stories.")
        return

    print_grouped(data, articles)

    by_id = {a["id"]: a for a in articles}

    # Always build and publish the digest page. Publishing is not a notification,
    # so it still happens during quiet hours — the agent keeps "collecting".
    print("\nBuilding the digest page...")
    published = publish_to_github_pages(generate_digest_html(data, articles))
    print(f"Digest saved to {DIGEST_PATH}")
    if published:
        print(f"Published to {PAGE_URL}")
    else:
        print("The page was saved locally but could not be published to GitHub.")

    # Quiet hours (23:00–08:00, Europe/Paris): keep silent — send nothing.
    if in_quiet_hours():
        print("\nQuiet hours (23h–8h, Europe/Paris) — no notifications sent.")
        return

    # The topic name is read from the environment / .env (never hard-coded).
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        print("No NTFY_TOPIC found — skipping phone notifications.")
        return

    # 1) Short mini/medium notifications. Only stories the AI marked "micro" or
    #    "medium" are pushed (the lead and "none" stories are not) — AND only if
    #    we haven't already notified you about them on an earlier run. Stories
    #    are remembered by their article link, which stays the same across runs.
    notified = load_notified()
    today_iso = now_local().strftime("%Y-%m-%d")

    candidates = [s for s in stories if s.get("notify") in ("micro", "medium")]
    fresh = []
    already_sent = 0
    for story in candidates:
        article = by_id.get(story["id"])
        link = article["link"] if article else None
        if link and link in notified:
            already_sent += 1  # pushed on a previous run — skip it
            continue
        fresh.append(story)

    print(f"\n{len(candidates)} mini/medium stories selected — "
          f"{already_sent} already sent before, {len(fresh)} new to send.")
    sent = 0
    for story in fresh:
        # A colored circle at the start of the title shows the tier at a glance:
        # 🔵 mini, 🟢 medium (🟣 is reserved for the digest notification below).
        article = by_id.get(story["id"])
        emoji = "🔵" if story.get("notify") == "micro" else "🟢"
        title = f"{emoji} {story['headline_fr']}"
        # Add the article's date after the title so you can see how recent it is.
        if article and article.get("date_str"):
            title = f"{title} ({article['date_str']})"
        try:
            send_ntfy_notification(topic, title, story["body_fr"])
            sent += 1
            # Remember it (by link) so future runs never send it again.
            if article and article.get("link"):
                notified[article["link"]] = today_iso
        except Exception as error:
            print(f"  (Could not send '{title}': {error})")
        # A short pause so we stay under ntfy's free rate limit.
        time.sleep(1)

    save_notified(notified)
    print(f"Sent {sent} of {len(fresh)} new mini/medium notifications.")

    # 2) Send the single "Résumé du jour" notification that opens the page.
    lead_headline = (data.get("lead") or {}).get("headline_fr", "")
    message = lead_headline or "Votre résumé du jour est prêt."
    message += f"\n\nTouchez pour ouvrir le résumé complet ({len(stories)} articles)."
    try:
        send_ntfy_notification(topic, "🟣 Résumé du jour", message, click=PAGE_URL)
        print('Sent the "Résumé du jour" notification.')
    except Exception as error:
        print(f"Could not send the digest notification: {error}")


if __name__ == "__main__":
    main()
