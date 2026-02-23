#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import zstandard
from tqdm.auto import tqdm


MEDIA_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "v.redd.it",
    "i.redd.it",
    "reddit.com/gallery",
    "imgur.com",
    "i.imgur.com",
    "gfycat.com",
    "redgifs.com",
    "streamable.com",
    "instagram.com",
    "tiktok.com",
    "vimeo.com",
}
MEDIA_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
)
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build filtered text-only dataset from opiates subreddit dumps."
    )
    parser.add_argument("--input-dir", default="../reddit/subreddits24")
    parser.add_argument("--subreddit", default="opiates")
    parser.add_argument("--output", default="data/opiates_text_filtered.csv")
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=600)
    return parser.parse_args()


def read_and_decode(
    reader, chunk_size: int, max_window_size: int, previous_chunk=None, bytes_read: int = 0
):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name: str):
    with open(file_name, "rb") as file_handle:
        buffer = ""
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line
            buffer = lines[-1]
        reader.close()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def has_media_url(text: str) -> bool:
    for match in URL_PATTERN.findall(text):
        lower = match.lower()
        if any(domain in lower for domain in MEDIA_DOMAINS):
            return True
        if lower.endswith(MEDIA_EXTENSIONS):
            return True
    return False


def is_media_submission(obj: dict, text: str) -> bool:
    if obj.get("is_video"):
        return True
    if obj.get("is_gallery"):
        return True
    if obj.get("media") is not None:
        return True
    if obj.get("media_metadata") is not None:
        return True
    if obj.get("post_hint") in {"image", "rich:video", "hosted:video", "link"}:
        return True
    url = (obj.get("url") or "").lower()
    if any(domain in url for domain in MEDIA_DOMAINS):
        return True
    if url.endswith(MEDIA_EXTENSIONS):
        return True
    if has_media_url(text):
        return True
    return False


def in_reasonable_range(text: str, min_words: int, max_words: int) -> tuple[bool, int]:
    wc = word_count(text)
    return min_words <= wc <= max_words, wc


def extract_text(obj: dict, kind: str) -> str:
    if kind == "submissions":
        selftext = obj.get("selftext") or ""
        if selftext in {"[removed]", "[deleted]"}:
            selftext = ""
        return selftext
    body = obj.get("body") or ""
    if body in {"[removed]", "[deleted]"}:
        body = ""
    return body


def main() -> None:
    args = parse_args()

    start_ts = int(datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    input_dir = Path(args.input_dir)

    counters = {
        "total_raw": 0,
        "json_error": 0,
        "missing_created_utc": 0,
        "before_start_date": 0,
        "empty_text": 0,
        "media_filtered": 0,
        "length_filtered": 0,
        "kept": 0,
    }
    records = []

    # for kind in ("submissions", "comments"):
    for kind in ("submissions",):
        file_path = input_dir / f"{args.subreddit}_{kind}.zst"
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            continue

        for line in tqdm(read_lines_zst(str(file_path)), desc=f"{args.subreddit}-{kind}", unit="rows"):
            counters["total_raw"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                counters["json_error"] += 1
                continue

            created_utc = obj.get("created_utc")
            if created_utc is None:
                counters["missing_created_utc"] += 1
                continue
            created_ts = int(created_utc)
            if created_ts < start_ts:
                counters["before_start_date"] += 1
                continue

            text = clean_text(extract_text(obj, kind))
            if not text:
                counters["empty_text"] += 1
                continue

            if kind == "submissions":
                media = is_media_submission(obj, text)
            else:
                media = has_media_url(text)
            if media:
                counters["media_filtered"] += 1
                continue

            ok_len, wc = in_reasonable_range(text, args.min_words, args.max_words)
            if not ok_len:
                counters["length_filtered"] += 1
                continue

            record = {
                "subreddit": args.subreddit,
                "kind": kind,
                "id": obj.get("id"),
                "author": obj.get("author"),
                "created_utc": created_ts,
                "created_dt": datetime.fromtimestamp(created_ts, tz=timezone.utc),
                "score": obj.get("score"),
                "text": text,
                "word_count": wc,
            }
            if kind == "submissions":
                record["title"] = clean_text(obj.get("title") or "")
                record["num_comments"] = obj.get("num_comments")
                record["permalink"] = obj.get("permalink")
            else:
                record["parent_id"] = obj.get("parent_id")
                record["link_id"] = obj.get("link_id")
            records.append(record)
            counters["kept"] += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out, index=False)

    print("\nDone.")
    print(f"Output: {out}")
    print("Counts:")
    for key, value in counters.items():
        print(f"  {key}: {value:,}")
    if counters["total_raw"] > 0:
        print(f"  keep_rate: {counters['kept'] / counters['total_raw']:.2%}")


if __name__ == "__main__":
    main()
