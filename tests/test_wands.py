from pathlib import Path

import pytest

from eval.wands import FILES, WandsError, document, fetch

ROW = {
    "product_id": "7",
    "product_name": " hair salon chair ",
    "product_class": "Massage Chairs",
    "category hierarchy": "Furniture / Office Furniture",
    "product_description": "x" * 1_200,
    "product_features": "color : black|weightcapacity:250||",
    "rating_count": "15.0",
    "average_rating": "4.5",
    "review_count": "",
}


def test_a_product_becomes_a_document_with_a_field_to_rerank_on():
    doc = document(ROW)

    assert doc["product_name"] == "hair salon chair"
    assert doc["product_features"] == "color: black; weightcapacity: 250"
    assert (doc["average_rating"], doc["review_count"]) == (4.5, None)
    name, category, description = doc["rerank_text"].split("\n")
    assert name == "hair salon chair"
    assert category == "Category: Furniture / Office Furniture / Massage Chairs"
    assert description == "x" * 1_000 + " ..."


def test_a_product_without_category_or_description_is_just_its_name():
    doc = document({**ROW, "product_class": "", "category hierarchy": "", "product_description": ""})

    assert doc["rerank_text"] == "hair salon chair"
    assert doc["product_class"] is None


def test_a_file_already_there_is_used_and_checked_not_downloaded(tmp_path: Path):
    file, header = FILES["product"]
    (tmp_path / file).write_text(header + "\n", encoding="utf-8")
    assert fetch("product", tmp_path) == tmp_path / file

    (tmp_path / file).write_text("something else\n", encoding="utf-8")
    with pytest.raises(WandsError):
        fetch("product", tmp_path)
