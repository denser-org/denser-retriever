from typing import Any, Dict
from denser_retriever.filter import generate_milvus_expr


def test_generate_milvus_expr_with_none_values() -> None:
    filter_dict: Dict[str, Any] = {
        "name": "John",
        "age": None,
        "city": None,
    }
    expected_expr = "name == 'John'"
    assert generate_milvus_expr(filter_dict) == expected_expr


def test_generate_milvus_expr_with_range_values() -> None:
    filter_dict: Dict[str, Any] = {
        "price": (10, 100),
        "rating": (4.5, 5.0),
    }
    expected_expr = (
        "price >= '10' and price <= '100' and rating >= '4.5' and rating <= '5.0'"
    )
    assert generate_milvus_expr(filter_dict) == expected_expr


def test_generate_milvus_expr_with_single_value() -> None:
    filter_dict: Dict[str, Any] = {
        "category": "electronics",
        "brand": "Apple",
    }
    expected_expr = "category == 'electronics' and brand == 'Apple'"
    assert generate_milvus_expr(filter_dict) == expected_expr


def test_generate_milvus_expr_with_empty_dict() -> None:
    filter_dict: Dict[str, Any] = {}
    expected_expr = ""
    assert generate_milvus_expr(filter_dict) == expected_expr
