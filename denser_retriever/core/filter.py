import time
from datetime import datetime
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


class FieldMapper:
    def __init__(self, fields: list[str]):
        self.original_to_key_map: Dict[str, str] = {}
        self.key_to_original_map: Dict[str, str] = {}
        self.field_types: Dict[str, str] = {}
        self.category_to_number_map = {}
        self.number_to_category_map = {}

        for f in fields:
            comps = f.split(":")
            assert len(comps) == 2 or len(comps) == 3

            if len(comps) == 3:
                original_key = comps[0]
                key = comps[1]
                field_type = comps[2]
            elif len(comps) == 2:
                original_key = key = comps[0]
                field_type = comps[1]

            self.original_to_key_map[original_key] = key
            self.key_to_original_map[key] = original_key
            self.field_types[key] = field_type

            if field_type == 'keyword':
                self.category_to_number_map[key] = {}
                self.number_to_category_map[key] = {}

    def get_key(self, original_key):
        return self.original_to_key_map.get(original_key)

    def get_keys(self) -> list[str]:
        return list(self.original_to_key_map.values())

    def get_original_key(self, key):
        return self.key_to_original_map.get(key)

    def get_original_keys(self) -> list[str]:
        return list(self.key_to_original_map.values())

    def get_field_type(self, key) -> str | None:
        return self.field_types.get(key)

    def convert_for_storage(self, data: Dict[str, Any]) -> Any:
        converted_data = {}
        for key, value in data.items():
            converted_key = self.get_key(key)
            if converted_key is None:
                continue
            converted_value = self.convert_query_condition(converted_key, value)
            converted_data[converted_key] = converted_value
        return converted_data[converted_key]

    def convert_to_original(self, data):
        original_data = {}
        for key, value in data.items():
            original_key = self.key_to_original_map.get(key)
            original_value = self.convert_back_to_original(key, value)
            original_data[original_key] = original_value
        return original_data[original_key]

    def convert_query_condition(self, key: str, value: str) -> Any:
        field_type = self.get_field_type(key)
        if field_type == "date":
            return self.convert_date_to_timestamp(value)
        elif field_type == "keyword":
            return self.convert_category_to_number(key, value)

    def convert_back_to_original(self, key, value):
        field_type = self.field_types.get(key)
        if field_type == 'date':
            return self.convert_timestamp_to_date(value)
        elif field_type == 'keyword':
            return self.convert_number_to_category(key, value)
        else:
            return value

    def convert_date_to_timestamp(self, date_str):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        timestamp = int(time.mktime(dt.timetuple()))
        return timestamp

    def convert_timestamp_to_date(self, timestamp):
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d")

    def convert_category_to_number(self, key, value):
        category_map = self.category_to_number_map[key]
        if value not in category_map:
            category_map[value] = len(category_map) + 1
        return category_map[value]

    def convert_number_to_category(self, key, number):
        return self.number_to_category_map[key].get(number)
