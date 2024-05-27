import json


def read_json_file(file_path):
    """
    Reads a JSON file and returns the JSON data.

    :param file_path: The path to the JSON file.
    :return: A Python dictionary or list with the JSON content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            return json_data

    except FileNotFoundError:
        print(f"No file found at {file_path}")
    except json.JSONDecodeError:
        print("Error decoding JSON data from the file")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
