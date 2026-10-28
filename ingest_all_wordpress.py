"""
Bulk WordPress Ingestion Script
================================
Fetches ALL posts from the WordPress REST API (with pagination),
skips posts already healthy in the DB, and ingests the rest.

Usage:
    python ingest_all_wordpress.py
    python ingest_all_wordpress.py --status draft        # include drafts
    python ingest_all_wordpress.py --per-page 50         # larger page size
    python ingest_all_wordpress.py --dry-run             # list only, no ingest
    python ingest_all_wordpress.py --from-page 3         # resume from page 3
"""

import argparse
import json
import logging
import sys
import time
import os

import requests

# Ensure imports from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from ingestion_url import URLIngestor

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bulk_ingest")


# ── WordPress API helpers ──────────────────────────────────────────────────

def build_auth():
    """Return (user, password) tuple if credentials are configured, else None."""
    u = config.WORDPRESS_USERNAME
    p = config.WORDPRESS_PASSWORD
    return (u, p) if u and p else None


def fetch_page(page: int, per_page: int, status: str, auth) -> tuple[list, int]:
    """
    Fetch one page of posts from the WP REST API.
    Returns (posts_list, total_pages).
    """
    url = config.WORDPRESS_API_URL          # e.g. https://kytisa.com/wp-json/wp/v2/posts
    params = {
        "page":     page,
        "per_page": per_page,
        "status":   status,
        "_fields":  "id,link,title,content,excerpt,date,modified,author,categories,tags,status",
    }
    resp = requests.get(url, params=params, auth=auth, timeout=30)

    # WP returns 400 when you request a page beyond the last one
    if resp.status_code == 400:
        return [], 0

    resp.raise_for_status()

    total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
    total_posts = int(resp.headers.get("X-WP-Total", 0))
    log.debug(f"Page {page}/{total_pages}  total_posts={total_posts}")

    return resp.json(), total_pages


# ── Main bulk ingestion ────────────────────────────────────────────────────

def run(args):
    auth = build_auth()
    if auth:
        log.info(f"✓ WordPress auth enabled  (user: {auth[0]})")
    else:
        log.warning("⚠  No WordPress credentials — only public posts will be fetched")

    ingestor = URLIngestor()

    # ── Counters ──────────────────────────────────────────────────────────
    stats = {
        "total_fetched": 0,
        "skipped_exists": 0,
        "ingested_ok":   0,
        "ingested_fail": 0,
        "low_quality":   0,
    }
    failed_posts = []

    try:
        page        = args.from_page
        total_pages = None          # unknown until first request

        while True:
            if total_pages is not None and page > total_pages:
                break

            log.info(f"── Fetching page {page}" +
                     (f"/{total_pages}" if total_pages else "") + " ──")

            try:
                posts, total_pages = fetch_page(
                    page, args.per_page, args.status, auth
                )
            except requests.HTTPError as e:
                log.error(f"HTTP error fetching page {page}: {e}")
                break
            except Exception as e:
                log.error(f"Unexpected error fetching page {page}: {e}")
                break

            if not posts:
                log.info("No posts returned — done.")
                break

            stats["total_fetched"] += len(posts)

            for post in posts:
                doc_id  = str(post.get("id", "?"))
                title   = post.get("title", {}).get("rendered", "(no title)")
                link    = post.get("link", "")

                # ── Pre-check: already in DB? ─────────────────────────────
                if ingestor.is_document_healthy(doc_id):
                    log.info(f"  [SKIP]    #{doc_id}  {title[:60]}")
                    stats["skipped_exists"] += 1
                    continue

                if args.dry_run:
                    log.info(f"  [DRY-RUN] #{doc_id}  {title[:60]}  {link}")
                    continue

                # ── Ingest ────────────────────────────────────────────────
                log.info(f"  [INGEST]  #{doc_id}  {title[:60]}")
                t0  = time.time()
                res = ingestor.ingest_post(post)
                dt  = time.time() - t0

                status_str = res.get("status", "?")
                msg        = res.get("message", "")
                chunks     = res.get("chunks_count", 0)

                if status_str == "success":
                    log.info(f"            ✓ {chunks} chunks  ({dt:.1f}s)")
                    stats["ingested_ok"] += 1
                elif status_str in ("exists", "already_indexed"):
                    log.info(f"            ↩ {msg}")
                    stats["skipped_exists"] += 1
                elif "quality" in msg.lower() or "short" in msg.lower():
                    log.warning(f"            ✗ low quality — {msg}")
                    stats["low_quality"] += 1
                else:
                    log.error(f"            ✗ FAILED — {msg}")
                    stats["ingested_fail"] += 1
                    failed_posts.append({"id": doc_id, "url": link, "error": msg})

                # Polite pause between ingestions
                time.sleep(args.delay)

            page += 1

    finally:
        ingestor.close()

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BULK INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Posts fetched   : {stats['total_fetched']}")
    print(f"  Already in DB   : {stats['skipped_exists']}")
    print(f"  Ingested OK     : {stats['ingested_ok']}")
    print(f"  Low quality     : {stats['low_quality']}")
    print(f"  Failed          : {stats['ingested_fail']}")
    print("=" * 60)

    if failed_posts:
        fail_file = "failed_posts.json"
        with open(fail_file, "w", encoding="utf-8") as f:
            json.dump(failed_posts, f, ensure_ascii=False, indent=2)
        print(f"\n⚠  {len(failed_posts)} failed posts saved to {fail_file}")

    if stats["ingested_fail"] > 0:
        sys.exit(1)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk-ingest all WordPress posts into GraphRAG"
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=20,
        help="Number of posts per API page (default: 20, max: 100)",
    )
    parser.add_argument(
        "--status",
        default="publish",
        help="WordPress post status to fetch (default: publish). "
             "Use 'any' for all statuses (requires auth).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between ingestions (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List posts that would be ingested without actually ingesting",
    )
    parser.add_argument(
        "--from-page",
        type=int,
        default=1,
        help="Start from this page number (useful to resume after failure)",
    )

    args = parser.parse_args()

    # Clamp per_page to WP API max
    if args.per_page > 100:
        log.warning("WP API max per_page is 100 — clamping.")
        args.per_page = 100

    run(args)
