import json
import warnings

import pandas as pd
import functools
import sys
from io import StringIO
from contextlib import contextmanager

def objects_to_dataframe(objects):
    # Convert a list of objects to a list of dictionaries
    data = [vars(obj) for obj in objects]
    # Create and return a DataFrame
    return pd.DataFrame(data)


def load_json_to_class(json_file_path, cls):
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure the JSON file has the correct structure (a list of objects)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("The JSON file must contain a list of objects.")

    # Extract attribute names from the class by instantiating an empty instance
    temp_instance = cls()
    class_attributes = temp_instance.__dict__.keys()

    # List to store instances of the class
    instances = []

    for obj_data in data:
        # Filter JSON data to only include keys matching class attributes
        filtered_data = {k: v for k, v in obj_data.items() if k in class_attributes}

        # Check for missing attributes and set them to None
        missing_attrs = set(class_attributes) - set(obj_data.keys())
        for missing in missing_attrs:
            filtered_data[missing] = None
            warnings.warn(f"Attribute '{missing}' is missing in JSON data and will be set to None.")

        # Create an instance of the class with the filtered data
        instance = cls(**filtered_data)
        instances.append(instance)

    return instances

def generate_class_from_json(json_file_path, class_name="GeneratedClass", to_file=False):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Ensure the JSON file has the correct structure (a list of objects)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("The JSON file must contain a list of objects.")

    # Get the keys of the first object to define the class attributes
    attributes = data[0].keys()

    # Generate class definition code
    class_code = f"class {class_name}:\n"
    class_code += "    def __init__(self, **kwargs):\n"

    # Add attributes to the class's __init__ method
    for attr in attributes:
        class_code += f"        self.{attr} = kwargs.get('{attr}', None)\n"

    if to_file:
        # Save the generated class to a .py file
        file_name = f"{class_name}.py"
        with open(file_name, "w") as file:
            file.write(class_code)
        print(f"Class code has been written to {file_name}")
    else:
        # Print the generated class code
        print(class_code)




@contextmanager
def suppress_output():
    """Context manager to temporarily suppress stdout."""
    old_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old_stdout

def suppress_prints(func):
    """
    Decorator that suppresses all print statements in the decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with suppress_output():
            return func(*args, **kwargs)
    return wrapper



# instances = load_json_to_class('train_spider_clean.json', SpiderDataset)
# for instance in instances:
#     print(instance)