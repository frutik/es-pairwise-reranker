"""Loads Wayfair's WANDS products (https://github.com/wayfair/WANDS, MIT licence, 42,994 furniture products) into an
Elasticsearch index, to try the reranker on.

    wands-load [--es http://localhost:9200] [--index wands] [--data-dir data/wands] [--limit N]

The product file (90 MB) is downloaded from GitHub the first time, into --data-dir, unless it is already there. The
index is dropped and made again, so the load can be re-run.

Each product keeps WANDS' fields, plus `rerank_text`: the field to point the text_similarity_reranker at. Elasticsearch
sends the judge only that one field. It joins the fields Quepid's WANDS experiment showed Jev (name, category, product
class, description), so the judge sees what it saw there.
"""

import argparse
import csv
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx

BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
FILES = {
    "product": (
        "product.csv",
        "product_id\tproduct_name\tproduct_class\tcategory hierarchy\tproduct_description\tproduct_features"
        "\trating_count\taverage_rating\treview_count",
    ),
    "query": ("query.csv", "query_id\tquery\tquery_class"),
    "label": ("label.csv", "id\tquery_id\tproduct_id\tlabel"),
}
MAX_BYTES = 200 * 1024 * 1024  # the product file is 90 MB; anything far bigger is not WANDS
BATCH = 2_000
# Descriptions run to thousands of characters; the judge is paid per token and Jev cuts a document at 4,000 anyway.
RERANK_DESCRIPTION_CHARS = 1_000

MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "product_id": {"type": "keyword"},
            "product_name": {"type": "text", "analyzer": "english"},
            "product_class": {"type": "text", "analyzer": "english", "fields": {"raw": {"type": "keyword"}}},
            "category_hierarchy": {"type": "text", "analyzer": "english", "fields": {"raw": {"type": "keyword"}}},
            "product_description": {"type": "text", "analyzer": "english"},
            "product_features": {"type": "text", "analyzer": "english"},
            "rating_count": {"type": "float"},
            "average_rating": {"type": "float"},
            "review_count": {"type": "float"},
            "rerank_text": {"type": "text", "analyzer": "english"},
        }
    },
}


class WandsError(Exception):
    pass


def fetch(name: str, data_dir: Path) -> Path:
    """The named WANDS file, downloaded first if it is not in data_dir. Written under another name and moved into place
    when whole, so a failed download never leaves half a file behind."""
    file, header = FILES[name]
    path = data_dir / file
    if path.exists() and path.stat().st_size > 0:
        check_header(path, header)
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(f".{os.getpid()}.part")
    print(f"downloading {BASE_URL}/{file}", file=sys.stderr)
    try:
        with httpx.stream("GET", f"{BASE_URL}/{file}", follow_redirects=True, timeout=60) as response:
            if response.status_code != 200:
                raise WandsError(f"GitHub answered {response.status_code} for {file}")
            written = 0
            with part.open("wb") as out:
                for chunk in response.iter_bytes():
                    written += len(chunk)
                    if written > MAX_BYTES:
                        raise WandsError(f"{file} is larger than {MAX_BYTES // 2**20} MB, so it is not WANDS'")
                    out.write(chunk)
        check_header(part, header)
        part.rename(path)
    finally:
        part.unlink(missing_ok=True)
    return path


def check_header(path: Path, header: str) -> None:
    with path.open(encoding="utf-8-sig") as file:
        first = file.readline().rstrip("\r\n")
    if first != header:
        raise WandsError(f"{path} does not start with WANDS' columns, so it was not read")


def rows(path: Path) -> Iterator[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file, delimiter="\t")


def document(row: dict) -> dict:
    name = row["product_name"].strip()
    product_class = row["product_class"].strip()
    category = row["category hierarchy"].strip()
    description = row["product_description"].strip()
    # "key:value|key : value|..." -> "key: value; key: value"
    features = "; ".join(
        ": ".join(part.strip() for part in feature.split(":", 1))
        for feature in row["product_features"].split("|")
        if feature.strip()
    )
    rerank = [name]
    if category or product_class:
        rerank.append("Category: " + " / ".join(part for part in (category, product_class) if part))
    if description:
        cut = description[:RERANK_DESCRIPTION_CHARS]
        rerank.append(cut + (" ..." if len(description) > len(cut) else ""))
    return {
        "product_id": row["product_id"],
        "product_name": name,
        "product_class": product_class or None,
        "category_hierarchy": category or None,
        "product_description": description or None,
        "product_features": features or None,
        "rating_count": _number(row["rating_count"]),
        "average_rating": _number(row["average_rating"]),
        "review_count": _number(row["review_count"]),
        "rerank_text": "\n".join(rerank),
    }


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def load(es: str, index: str, data_dir: Path, limit: int | None = None, out=sys.stdout) -> int:
    path = fetch("product", data_dir)
    with httpx.Client(base_url=es, timeout=120) as client:
        client.delete(f"/{index}", params={"ignore_unavailable": "true"})
        response = client.put(f"/{index}", json=MAPPING)
        if response.status_code != 200:
            raise WandsError(f"could not create the index: {response.status_code} {response.text[:300]}")
        loaded = 0
        batch: list[str] = []
        for number, row in enumerate(rows(path)):
            if limit is not None and number >= limit:
                break
            doc = document(row)
            batch += [json.dumps({"index": {"_id": doc["product_id"]}}), json.dumps(doc, ensure_ascii=False)]
            if len(batch) >= 2 * BATCH:
                loaded += _bulk(client, index, batch)
                batch = []
                print(f"\r{loaded} products", end="", file=out, flush=True)
        if batch:
            loaded += _bulk(client, index, batch)
        client.post(f"/{index}/_refresh")
    print(f"\r{loaded} products in {es}/{index}", file=out)
    return loaded


def _bulk(client: httpx.Client, index: str, lines: list[str]) -> int:
    response = client.post(
        f"/{index}/_bulk", content="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"}
    )
    if response.status_code != 200:
        raise WandsError(f"bulk load failed: {response.status_code} {response.text[:300]}")
    body = response.json()
    if body.get("errors"):
        failed = next(item["index"] for item in body["items"] if "error" in item["index"])
        raise WandsError(f"bulk load failed for product {failed['_id']}: {failed['error']}")
    return len(body["items"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--es", default=os.environ.get("ES_URL", "http://localhost:9200"), help="Elasticsearch URL")
    parser.add_argument("--index", default="wands")
    parser.add_argument("--data-dir", type=Path, default=Path("data/wands"), help="where the WANDS files are kept")
    parser.add_argument("--limit", type=int, help="only the first this many products")
    arguments = parser.parse_args(argv)
    try:
        load(arguments.es, arguments.index, arguments.data_dir, arguments.limit)
    except (WandsError, httpx.HTTPError) as error:
        sys.exit(f"wands-load: {error}")


if __name__ == "__main__":
    main()
