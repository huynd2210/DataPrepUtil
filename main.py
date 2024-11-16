import yaml

from core.data_loader import load_spider, get_spider_db_path
from core.sql_tools import getDatabaseSchemaForPrompt

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

import sqlite3
import pandas as pd
import ollama

from models.SpiderDataset import SpiderDataset
from models.SQLEvaluationEntry import SQLEvaluationEntry
from core.utils import suppress_prints, objects_to_dataframe, load_json_to_class

def prompt(model_name, promptTemplate=config["prompt_template"], **kwargs):
    return ollama.generate(model=model_name, prompt=promptTemplate.format(**kwargs))

def generateSQL(model_name, promptTemplate=config["prompt_template"], db_path="", **kwargs):
    # kwargs should include arguments for the prompt template
    kwargs["db_path"] = db_path
    return prompt(model_name, promptTemplate=promptTemplate, **kwargs)

def generateSQLEvaluationEntry(model_name: str, spider_dataset_entry: SpiderDataset):
    request = spider_dataset_entry.question
    schema = getDatabaseSchemaForPrompt(get_spider_db_path(spider_dataset_entry.db_id))

    generated_sql = generateSQL(model_name=model_name,
                                db_path=get_spider_db_path(spider_dataset_entry.db_id),
                                request=request,
                                schema=schema
                                )

    return SQLEvaluationEntry(
        get_spider_db_path(spider_dataset_entry.db_id),
        generated_sql,
        spider_dataset_entry.query,
        spider_dataset_entry.question
    )

# @suppress_prints
def evaluateSQLGenerationEntry(evaluation_entry: SQLEvaluationEntry, conn=None, close_conn=True) -> SQLEvaluationEntry:
    """
    Evaluate a generated SQL query against a gold (reference) SQL query by executing both
    on the specified SQLite database and comparing their results.

    Parameters:
    - test_case (SQLTestCase): The test case containing database path and SQL queries.

    Returns:
    - bool: True if both queries return identical results, False otherwise.
    """
    print("Question: " + evaluation_entry.question)
    print("Gold SQL: " + evaluation_entry.gold_sql)
    print("Generated SQL: " + evaluation_entry.generated_sql)

    try:
        # Connect to the database
        if conn is None:
            conn = sqlite3.connect(evaluation_entry.db_path)

        # Execute both queries and fetch results as dataframes
        generated_df = pd.read_sql_query(evaluation_entry.generated_sql, conn)
        gold_df = pd.read_sql_query(evaluation_entry.gold_sql, conn)

        if close_conn:
            conn.close()

        # Check if both dataframes are identical
        if generated_df.equals(gold_df):
            print("The generated SQL matches the gold SQL.")
            evaluation_entry.isCorrect = True
            return evaluation_entry
        else:
            print("The generated SQL does not match the gold SQL.")

            # Show differences
            diff = pd.concat([generated_df, gold_df]).drop_duplicates(keep=False)
            print("Differences:")
            print(diff)

            evaluation_entry.isCorrect = False
            return evaluation_entry

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

def evaluateModel(model_name: str, dataset="spider", split="train"):
    result = []
    if dataset == "spider":
        spider_instances = load_spider(split)
        for instance in spider_instances:

            evaluation_entry = generateSQLEvaluationEntry(model_name, instance)
            result.append(evaluateSQLGenerationEntry(evaluation_entry))
            break

    pd = objects_to_dataframe(result)
    return pd

def analyseEvaluation(evaluationResultDf: pd.DataFrame):
    # Extract the 'isCorrect' attribute from each SQLEvaluationEntry instance
    evaluationResultDf['isCorrect'] = evaluationResultDf[0].apply(lambda entry: entry.isCorrect)

    # Calculate the percentage of correct entries (isCorrect == True)
    percentage_correct = evaluationResultDf['isCorrect'].sum() / len(evaluationResultDf) * 100

    print(f"Percentage of correct entries: {percentage_correct:.2f}%")


# print(response)

# load_spider("train")
result = evaluateModel("llama3.1:latest")
analyseEvaluation(result)