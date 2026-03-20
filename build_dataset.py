#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ClaimTaxo-ready ICE dataset from posts/comments JSONL."
    )
    p.add_argument(
        "--posts",
        default="/home/yli62/Documents/r_ICE_Raids_posts.jsonl",
        help="Path to posts JSONL file.",
    )
    p.add_argument(
        "--comments",
        default="/home/yli62/Documents/r_ICE_Raids_comments.jsonl",
        help="Path to comments JSONL file.",
    )
    p.add_argument("--output", default="data/ice_raids_text_structured.csv")
    p.add_argument("--subreddit", default="ICE_Raids")
    p.add_argument("--start-date", default="2014-01-01")
    return p.parse_args()


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def to_dt(created_utc: int | float | str | None):
    if created_utc is None:
        return None
    try:
        ts = int(float(created_utc))
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def post_text_block(selftext: str) -> str:
    return f"[POST_TEXT]\n{clean_text(selftext)}"


def comment_text_block(selftext: str, body: str) -> str:
    return (
        f"[POST_TEXT]\n{clean_text(selftext)}\n\n"
        f"[TOP_LEVEL_COMMENT]\n{clean_text(body)}"
    )


def main() -> None:
    args = parse_args()
    posts_path = Path(args.posts)
    comments_path = Path(args.comments)

    start_ts = int(datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    counters = {
        "posts_read": 0,
        "comments_read": 0,
        "comment_missing_parent_post": 0,
        "comment_not_top_level": 0,
        "rows_written": 0,
    }

    posts_by_id = {}
    rows = []

    for post in read_jsonl(posts_path):
        counters["posts_read"] += 1
        post_id = (post.get("id") or "").strip()
        created_utc = post.get("created_utc")
        if not post_id or created_utc is None:
            continue
        try:
            created_ts = int(float(created_utc))
        except (TypeError, ValueError):
            continue
        if created_ts < start_ts:
            continue

        title = clean_text(post.get("title") or "")
        selftext = post.get("selftext") or ""
        posts_by_id[post_id] = {
            "id": post_id,
            "author": post.get("author"),
            "score": post.get("score"),
            "created_utc": created_ts,
            "created_dt": to_dt(created_ts),
            "title": title,
            "selftext": selftext,
            "num_comments": post.get("num_comments"),
            "permalink": post.get("permalink"),
        }

        rows.append(
            {
                "subreddit": args.subreddit,
                "kind": "submissions",
                "id": post_id,
                "post_id": post_id,
                "comment_id": "",
                "author": post.get("author"),
                "created_utc": created_ts,
                "created_dt": to_dt(created_ts),
                "score": post.get("score"),
                "title": title,
                "text": post_text_block(selftext),
                "num_comments": post.get("num_comments"),
                "permalink": post.get("permalink"),
                "parent_id": "",
                "link_id": "",
            }
        )
        counters["rows_written"] += 1

    for comment in read_jsonl(comments_path):
        counters["comments_read"] += 1
        comment_id = (comment.get("id") or "").strip()
        parent_id = (comment.get("parent_id") or "").strip()
        link_id = (comment.get("link_id") or "").strip()
        created_utc = comment.get("created_utc")
        if not comment_id or created_utc is None:
            continue
        try:
            created_ts = int(float(created_utc))
        except (TypeError, ValueError):
            continue
        if created_ts < start_ts:
            continue

        # Keep top-level comments only.
        if not parent_id.startswith("t3_"):
            counters["comment_not_top_level"] += 1
            continue

        post_id = link_id.replace("t3_", "", 1) if link_id.startswith("t3_") else ""
        if not post_id:
            post_id = parent_id.replace("t3_", "", 1) if parent_id.startswith("t3_") else ""
        parent_post = posts_by_id.get(post_id)
        if parent_post is None:
            counters["comment_missing_parent_post"] += 1
            continue

        rows.append(
            {
                "subreddit": args.subreddit,
                "kind": "submission_top_comment",
                "id": f"{post_id}_{comment_id}",
                "post_id": post_id,
                "comment_id": comment_id,
                "author": comment.get("author"),
                "created_utc": created_ts,
                "created_dt": to_dt(created_ts),
                "score": comment.get("score"),
                "title": parent_post["title"],
                "text": comment_text_block(parent_post["selftext"], comment.get("body") or ""),
                "num_comments": parent_post.get("num_comments"),
                "permalink": comment.get("permalink") or parent_post.get("permalink"),
                "parent_id": parent_id,
                "link_id": link_id,
            }
        )
        counters["rows_written"] += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print("Done.")
    print(f"Output: {out}")
    for k, v in counters.items():
        print(f"{k}: {v:,}")


if __name__ == "__main__":
    main()
