import json
import warnings
from collections import Counter

import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
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

def cleanLLMResponse(response, openTag="<final answer>", closeTag="</final answer>"):
    response = response.replace("```sql", "").replace("\n", " ").strip()

    if openTag in response:
        response = response.split(openTag)[1].strip()

    if closeTag in response:
        response = response.split(closeTag)[0].strip()

    return response

def compareCountersWithDiff(count1, count2):
    if count1 == count2:
        return {
            "equal": True,
            "diff_in_list1": {},
            "diff_in_list2": {}
        }
    else:
        # Find extra elements in list1
        diff1 = count1 - count2
        # Find extra elements in list2
        diff2 = count2 - count1

        return {
            "equal": False,
            "diff_in_list1": dict(diff1),
            "diff_in_list2": dict(diff2)
        }

# @suppress_prints
def isDataFrameEqual(df1, df2):
    df1 = df1.copy()
    df2 = df2.copy()
    df1Dict = df1.to_dict(orient='index')
    df2Dict = df2.to_dict(orient='index')

    df1Values = [list(row.values()) for row in df1Dict.values()]
    df2Values = [list(row.values()) for row in df2Dict.values()]

    for i in range(len(df1Values)):
        counter1 = Counter(df1Values[i])
        counter2 = Counter(df2Values[i])
        comparison = compareCountersWithDiff(counter1, counter2)

        if not comparison["equal"]:
            print("-----Differences in row " + str(i) + "-----")
            print(comparison)
            return False
    return True


def compare_dataframes(df1, df2, ignore_index=True, ignore_column_order=True):
    """
    Compare two DataFrames while ignoring row order and optionally column order.

    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        The DataFrames to compare
    ignore_index : bool, default=True
        Whether to ignore index values during comparison
    ignore_column_order : bool, default=True
        Whether to ignore column order during comparison

    Returns:
    --------
    bool
        True if DataFrames are equal (ignoring specified ordering), False otherwise
    dict
        Dictionary containing detailed comparison results
    """
    # Make copies to avoid modifying original DataFrames
    df1 = df1.copy()
    df2 = df2.copy()

    results = {
        'are_equal': False,
        'shape_match': False,
        # 'column_match': False,
        'data_match': False,
        'differences': []
    }

    # Check shapes
    if df1.shape != df2.shape:
        results['differences'].append(f"Shape mismatch: {df1.shape} vs {df2.shape}")
        return False, results
    results['shape_match'] = True

    # # Check columns
    # if set(df1.columns) != set(df2.columns):
    #     results['differences'].append("Column names don't match")
    #     return False, results
    # results['column_match'] = True

    # Sort columns if ignore_column_order is True
    if ignore_column_order:
        df1 = df1.reindex(sorted(df1.columns), axis=1)
        df2 = df2.reindex(sorted(df2.columns), axis=1)

    # Reset index if ignore_index is True
    if ignore_index:
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

    # Sort values by all columns to ignore row order
    df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
    df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

    # Compare data
    if df1.equals(df2):
        results['are_equal'] = True
        results['data_match'] = True
        return True, results

    return False, results


def print_comparison_results(comparison_results):
    """
    Print the comparison results in a readable format.

    Parameters:
    -----------
    comparison_results : tuple
        Tuple containing (bool, dict) returned by compare_dataframes
    """
    is_equal, results = comparison_results

    print("DataFrame Comparison Results:")
    print("-" * 30)
    print(f"Overall Equal: {is_equal}")
    print(f"Shape Match: {results['shape_match']}")
    # print(f"Column Match: {results['column_match']}")
    print(f"Data Match: {results['data_match']}")

# instances = load_json_to_class('train_spider_clean.json', SpiderDataset)
# for instance in instances:
#     print(instance)