import csv
import json
import logging
import datetime
import random

logger = logging.getLogger(__name__)

random.seed(10)


def csv_to_jsonl(csv_filepath, jsonl_filepath, key_update):
    """
    Convert a CSV file to a JSON Lines (JSONL) file.

    Args:
    csv_filepath (str): The path to the input CSV file.
    jsonl_filepath (str): The path to the output JSONL file.
    """

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
                age_str = item["Age"]
                age = int(float(age_str)) if len(age_str) > 0 else 0
                item["Birthday"] = generate_birthday(age)
                if item['source']:
                    jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    logger.info(f"Skip {item}")


def generate_birthday(age):
    # Calculate the birth year
    birth_year = 1912 - age

    # Generate a random day and month
    try:
        # Generate a random month and a random day in that month
        random_month = random.randint(1, 12)
        # Handle different days in months
        if random_month in (1, 3, 5, 7, 8, 10, 12):
            random_day = random.randint(1, 31)
        elif random_month == 2:
            # Check for leap year
            if (birth_year % 4 == 0 and birth_year % 100 != 0) or (birth_year % 400 == 0):
                random_day = random.randint(1, 29)
            else:
                random_day = random.randint(1, 28)
        else:
            random_day = random.randint(1, 30)

        # Create the random birth date in the birth year
        birthday = datetime.date(birth_year, random_month, random_day)
        # date.strftime("%Y-%m-%d")
        return birthday.strftime("%Y-%m-%d")
    except ValueError:
        # In case of a date error, we retry the function
        return generate_birthday(age)


if __name__ == "__main__":
    input_file = "tests/test_data/titanic.csv"
    output_file = "tests/test_data/titanic_top10.jsonl"
    key_update = {"PassengerId": "source", "Title": "title", "Name": "text"}
    csv_to_jsonl(input_file, output_file, key_update)
