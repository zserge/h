#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from sentence_transformers import SentenceTransformer, util

FEED_URLS = [
    "https://news.ycombinator.com/rss",
    "https://www.reddit.com/r/technology/.rss",
    "https://www.theverge.com/rss/index.xml",
    "https://lwn.net/headlines/newrss",
    "https://feeds.arstechnica.com/arstechnica/index/",
    "https://hackaday.com/blog/feed/",
    "https://www.theguardian.com/uk/rss",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://newsfeed.zeit.de/index",
    "https://finance.yahoo.com/news/rssindex",
]

BLACKLIST = [
    "donald trump",
    "elon musk",
    "articles about US politics, Trump, Musk, other billionaires or American political debates",
    "news covering crypto markets, NFTs, or blockchain startups",
    "anything about Elon Musk, Twitter, Facebook or other social networks",
    "american football and NFL, basketball and NBA, baseball and MLB",
    "netflix original shows and actors, hollywood movies, celebrity gossip",
]

# model = SentenceTransformer("all-MiniLM-L6-v2")
# model = SentenceTransformer("BAAI/bge-base-en-v1.5")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

BLACKLIST_EMB = model.encode(["Represent this sentence for retrieval: " + b for b in BLACKLIST], normalize_embeddings=True)

def fetch(url):
    try:
        print(f"[INFO] Fetching: {url}")
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read()
    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return None

def date(s, rfc822=False):
    try:
        dt = parsedate_to_datetime(s) if rfc822 else datetime.fromisoformat(s.rstrip("Z"))
    except Exception as e:
        print(f"[ERROR] Date parse failed: {date_str} {e}")
        return None
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt

def parse(xml):
    try:
        root = ET.fromstring(xml)
    except Exception as e:
        print(f"[ERROR] XML parse failed: {e}")
        return []
    ns = "{http://www.w3.org/2005/Atom}"
    entries = []
    if root.tag.endswith("feed"):
        for e in root.findall(f"{ns}entry"):
            entries.append({
                "title": (e.findtext(f"{ns}title") or "(no title)").strip(),
                "link": (e.find(f"{ns}link").attrib.get("href") if e.find(f"{ns}link") is not None else "#").strip(),
                "summary": (e.findtext(f"{ns}summary") or e.findtext(f"{ns}content") or "").strip(),
                "published": date(e.findtext(f"{ns}updated") or "", False)
            })
    else:
        for e in root.findall(".//item"):
            entries.append({
                "title": (e.findtext("title") or "(no title)").strip(),
                "link": (e.findtext("link") or "#").strip(),
                "summary": (e.findtext("description") or "").strip(),
                "published": date(e.findtext("pubDate") or "", True)
            })
    print(f"[INFO] Found {len(entries)} entries.")
    return entries

def blacklist(entry):
    text = f"{entry['title']} {entry['summary']}".lower()
    if any(k.lower() in text for k in BLACKLIST):
        print(f"[BLOCK] Keyword ban: {entry['title']}")
        return True
    embedding = model.encode(text, normalize_embeddings=True)
    cosine_scores = util.cos_sim(embedding, BLACKLIST_EMB)
    max_score = cosine_scores.max().item()
    if max_score > 0.6:
        print(f"[BLOCK] Semantic ban ({max_score}): {entry['title']}")
        return True
    return False

def domain(url):
    try: return urlparse(url).netloc.replace("www.", "").lower()
    except: return "?"

def group(entries):
    today = datetime.now().astimezone().date()
    week_ago = today - timedelta(days=7)
    groups = {"Today": [], "This Week": []}
    for e in entries:
        if not e["published"] or e["published"].date() <= week_ago or blacklist(e): continue
        if e["published"].date() == today: groups["Today"].append(e)
        elif week_ago < e["published"].date() < today: groups["This Week"].append(e)
    return groups

def render(groups):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = ["<html><head><meta charset='utf-8'><title>News</title><link rel='stylesheet' href='styles.css'></head><body>"]
    for section in ["Today", "This Week"]:
        items = groups.get(section, [])
        if not items: continue
        html.append(f"<h2>{section}</h2><ul>")
        for e in sorted(items, key=lambda x: x["published"], reverse=True):
            html.append(f'<li><a href="{e["link"]}">{e["title"]}<em>({domain(e["link"])})</em></a></li>')
        html.append("</ul>")
    html.append(f"<footer><p>Updated {now}</p></footer></body></html>")
    print(f"[OK] Rendered HTML with {sum(len(v) for v in groups.values())} entries.")
    return html


if __name__ == "__main__":
    raw = [fetch(url) for url in FEED_URLS]
    parsed = [parse(xml) for xml in raw if xml]
    grouped = group([item for sub in parsed for item in sub])
    html = render(grouped)
    Path("index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"[OK] Wrote index.html.")

