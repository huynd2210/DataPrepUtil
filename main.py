import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

import sqlite3
import pandas as pd
import ollama

from models.SpiderDataset import SpiderDataset
from models.SQLEvaluationEntry import SQLEvaluationEntry
from utils import suppress_prints, objects_to_dataframe, load_json_to_class


def get_spider_db_path(db_id, spiderRootPath=config["spider_root_path"], split="train"):
    if split == "train":
        return f"{spiderRootPath}/database/{db_id}/{db_id}.sqlite"
    else:
        return f"{spiderRootPath}/test_database/{db_id}/{db_id}.sqlite"

def load_spider(split: str = "train") -> list[SpiderDataset]:
    if split == "train":
        instances = load_json_to_class('old/train_spider_clean.json', SpiderDataset)
        for instance in instances:
            print(instance)
        return instances

def prompt(model_name, promptTemplate=config["prompt_template"], **kwargs):
    return ollama.generate(model=model_name, prompt=promptTemplate.format(**kwargs))

def generateSQL(model_name, promptTemplate=config["prompt_template"], db_path="", **kwargs):
    kwargs["db_path"] = db_path
    return prompt(model_name, promptTemplate=promptTemplate, **kwargs)

def generateSQLEvaluationEntry(generated_sql, spider_dataset_entry: SpiderDataset):
    return SQLEvaluationEntry(
        get_spider_db_path(spider_dataset_entry.db_id),
        generated_sql,
        spider_dataset_entry.query,
        spider_dataset_entry.question
    )

# def generateSQL(db_path, )


def generateSQLEntry(model_name: str, dataset="spider", split="train"):
    if dataset == "spider":
        instances = load_spider(split)
        for instance in instances:
            # response = ollama.generate(model='llama3.1', prompt='Why is the sky blue?')

            evaluation_entry = generateSQLEvaluationEntry(instance.generated_sql, instance)
            return evaluation_entry



# @suppress_prints
def evaluateSQLGenerationEntry(evaluation_entry: SQLEvaluationEntry) -> SQLEvaluationEntry:
    """
    Evaluate a generated SQL query against a gold (reference) SQL query by executing both
    on the specified SQLite database and comparing their results.

    Parameters:
    - test_case (SQLTestCase): The test case containing database path and SQL queries.

    Returns:
    - bool: True if both queries return identical results, False otherwise.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(evaluation_entry.db_path)

        # Execute both queries and fetch results as dataframes
        generated_df = pd.read_sql_query(evaluation_entry.generated_sql, conn)
        gold_df = pd.read_sql_query(evaluation_entry.gold_sql, conn)

        # Close the database connection
        conn.close()

        # Check if both dataframes are identical
        if generated_df.equals(gold_df):
            print("The generated SQL matches the gold SQL.")
            evaluation_entry.result = True
            return evaluation_entry
        else:
            print("The generated SQL does not match the gold SQL.")

            # Show differences
            diff = pd.concat([generated_df, gold_df]).drop_duplicates(keep=False)
            print("Differences:")
            print(diff)

            evaluation_entry.result = False
            return evaluation_entry

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


def evaluateModel(model_name: str, dataset="spider", split="train"):
    result = []
    if dataset == "spider":
        instances = load_spider(split)
        for instance in instances:
            evaluation_entry = generateSQLEvaluationEntry(instance.generated_sql, instance)
            result.append(evaluateSQLGenerationEntry(evaluation_entry))
    objects_to_dataframe(result)





# print(response)

# load_spider("train")
