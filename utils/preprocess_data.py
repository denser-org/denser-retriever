import csv
import json
import logging

logger = logging.getLogger(__name__)


def csv_to_jsonl(csv_filepath, jsonl_filepath):
    """
    Convert a CSV file to a JSON Lines (JSONL) file.

    Args:
    csv_filepath (str): The path to the input CSV file.
    jsonl_filepath (str): The path to the output JSONL file.
    """
    key_update = {"原始链接": "source", "案件名称": "title", "全文": "text"}
    with open(csv_filepath, mode='r', newline='', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        with open(jsonl_filepath, mode='w', newline='', encoding='utf-8') as jsonl_file:
            # Convert each row to JSON and write it to the JSONL file
            for item in csv_reader:
                for key in key_update.keys():
                    if key in item:
                        item[key_update[key]] = item.pop(key)
                    else:
                        item[key_update[key]] = ""
                        logger.info(f"Missing key {key} in {item}")
                item["pid"] = -1
                if item['source']:
                    jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    logger.info(f"Skip {item}")


if __name__ == "__main__":
    input_file = "tests/test_data/cpws_2021_10_top10.csv"
    output_file = "tests/test_data/cpws_passages_top10.jsonl"
    csv_to_jsonl(input_file, output_file)
