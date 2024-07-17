from typing import Any, Dict


def generate_milvus_expr(filter_dict: Dict[str, Any]) -> str:
    """Generate a Milvus expression from a filter dictionary."""
    expressions = []
    for key, value in filter_dict.items():
        if value is None:
            continue
        if isinstance(value, tuple) and len(value) == 2:
            start, end = value
            expressions.append(f"{key} >= '{start}' and {key} <= '{end}'")
        else:
            expressions.append(f"{key} == '{value}'")
    return " and ".join(expressions)
