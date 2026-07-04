import json
import os


def ensure_directory(path: str):
    """
    Creates a directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def save_json(data, file_path: str):
    """
    Saves Python object as JSON.
    """
    ensure_directory(os.path.dirname(file_path))

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def load_json(file_path: str):
    """
    Loads JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_exists(path: str):
    return os.path.exists(path)


def delete_file(path: str):
    if os.path.exists(path):
        os.remove(path)
