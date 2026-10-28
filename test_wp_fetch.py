"""
test_wp_fetch.py  — Revised
----------------------------
1. Validates WordPress API credentials from .env
2. Fetches the full article content via the WP REST API
3. Shows content length + preview
4. Force-ingests into GraphRAG (Neo4j + Qdrant)

Usage:
    python test_wp_fetch.py
"""
import sys, os, re, json, traceback, requests
from html.parser import HTMLParser
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

WP_USER     = os.getenv("WORDPRESS_USERNAME", "")
WP_PASS     = os.getenv("WORDPRESS_PASSWORD", "")   # Application Password (spaces OK)
WP_API      = os.getenv("WORDPRESS_API_URL", "https://kytisa.com/wp-json/wp/v2/posts")
TARGET_URL  = "https://kytisa.com/code-general-des-impots-2026-lfi-26"

auth    = (WP_USER, WP_PASS) if WP_USER and WP_PASS else None
headers = {"User-Agent": "Mozilla/5.0 (compatible; WPIngest/1.0)"}


# ── Minimal HTML stripper ────────────────────────────────────────────────────
class _Strip(HTMLParser):
    def __init__(self): super().__init__(); self.parts = []
    def handle_data(self, d):
        t = d.strip()
        if t: self.parts.append(t)

def strip_html(html_str: str) -> str:
    p = _Strip(); p.feed(html_str)
    return "\n".join(p.parts)


# ── Step 1: Test credentials directly against the API ────────────────────────
print("=" * 65)
print("  WordPress API Credential Test")
print("=" * 65)
print(f"  API URL  : {WP_API}")
print(f"  Username : {WP_USER}")
print(f"  Password : {'*' * len(WP_PASS) if WP_PASS else '(not set)'}")
print(f"  Target   : {TARGET_URL}")
print()

print("[1] Testing credentials against WP JSON API root...")
try:
    base_domain = re.match(r'(https?://[^/]+)', TARGET_URL).group(1)
    test_url = f"{base_domain}/wp-json/wp/v2/users/me"
    r = requests.get(test_url, auth=auth, headers=headers, timeout=20)
    if r.status_code == 200:
        me = r.json()
        print(f"    ✓ Auth OK — Logged in as: {me.get('name', '?')} (role: {me.get('roles', ['?'])})")
    elif r.status_code == 401:
        print(f"    ✗ Auth FAILED (401 Unauthorized) — check username/password")
        print(f"      Response: {r.text[:200]}")
    else:
        print(f"    ⚠ Unexpected status {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"    ✗ Connection error: {e}")


# ── Step 2: Fetch page HTML to discover exact post API link ──────────────────
print(f"\n[2] Fetching page HTML to discover post API link...")
try:
    resp = requests.get(TARGET_URL, auth=auth, headers=headers, timeout=30)
    print(f"    HTTP {resp.status_code}  |  {len(resp.text):,} HTML chars")
    resp.raise_for_status()
    html = resp.text
except Exception as e:
    print(f"    FAILED: {e}"); sys.exit(1)

# Discover API link
post_api_url = None
m = re.search(
    r'type=["\']application/json["\'] href=["\']'
    r'(https?://[^"\']+/wp-json/wp/v2/posts/\d+)["\']',
    html
)
if m:
    post_api_url = m.group(1)
    print(f"    ✓ Discovered post API: {post_api_url}")
else:
    slug = TARGET_URL.rstrip("/").split("/")[-1]
    base = re.match(r'(https?://[^/]+)', TARGET_URL).group(1)
    post_api_url = f"{base}/wp-json/wp/v2/posts?slug={slug}"
    print(f"    ⚠ No link tag found — falling back to slug search: {post_api_url}")


# ── Step 3: Fetch the post JSON ──────────────────────────────────────────────
print(f"\n[3] Fetching post JSON...")
try:
    api_r = requests.get(post_api_url, auth=auth, headers=headers, timeout=30)
    print(f"    HTTP {api_r.status_code}")
    api_r.raise_for_status()
    post_data = api_r.json()
except Exception as e:
    print(f"    FAILED: {e}"); sys.exit(1)

if isinstance(post_data, list):
    if not post_data:
        print("    ERROR: API returned empty list — slug not found"); sys.exit(1)
    post_data = post_data[0]


# ── Step 4: Show article details ─────────────────────────────────────────────
post_id       = post_data.get("id")
post_title    = post_data.get("title", {}).get("rendered", "(no title)")
content_html  = post_data.get("content", {}).get("rendered", "")
excerpt_html  = post_data.get("excerpt", {}).get("rendered", "")
post_link     = post_data.get("link", TARGET_URL)
post_date     = post_data.get("date", "?")
post_modified = post_data.get("modified", "?")
post_status   = post_data.get("status", "?")

content_text = strip_html(content_html)

print(f"\n[4] Article details:")
print(f"    ID       : {post_id}")
print(f"    Status   : {post_status}")
print(f"    Date     : {post_date}  |  Modified: {post_modified}")
print(f"    Title    : {post_title}")
print(f"    URL      : {post_link}")
print(f"    Content  : {len(content_text):,} chars  ({len(content_html):,} HTML chars)")
print()

if len(content_text) < 500:
    print("    ⚠ WARNING: Content looks truncated — may be paywalled or auth issue")
    print(f"\n    Full content:\n{content_text}")
else:
    print(f"    ✓ Content looks complete ({len(content_text):,} chars)")
    print(f"\n    --- First 1000 chars ---")
    print(content_text[:1000])
    print(f"\n    --- Last 500 chars ---")
    print(content_text[-500:])


# ── Step 5: Force ingest into GraphRAG ───────────────────────────────────────
print(f"\n{'=' * 65}")
print("[5] Force-ingesting into GraphRAG (Neo4j + Qdrant)...")
print(f"{'=' * 65}")

try:
    from ingestion_url import URLIngestor
    ingestor = URLIngestor()
    result = ingestor.ingest_url(TARGET_URL, force=True)
    ingestor.close()
    print()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    status = result.get("status", "?")
    chunks = result.get("chunks_count", 0)
    if status == "success":
        print(f"\n    ✓ Ingestion SUCCEEDED — {chunks} chunks indexed")
    elif status == "exists":
        print(f"\n    ✓ Already indexed ({chunks} chunks)")
    else:
        print(f"\n    ✗ Ingestion FAILED: {result.get('message', '?')}")
except ImportError as e:
    print(f"    URLIngestor not available: {e}")
except Exception as e:
    print(f"    Ingestion error: {e}")
    traceback.print_exc()
