import sqlite3

import pandas as pd
from tqdm import tqdm

from core.data_loader import load_spider
from core.generation import generateSQLEvaluationEntry
from core.utils import isDataFrameEqual, objects_to_dataframe, loadToObjectsFromFile, config
from models.SQLEvaluationEntry import SQLEvaluationEntry

def evaluateSQL(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    try:
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    return set(predicted_res) == set(ground_truth_res)

def evaluateSQLGenerationEntry(evaluation_entry: SQLEvaluationEntry, conn=None, close_conn=True) -> SQLEvaluationEntry:
    """
    Evaluate a generated SQL query against a gold (reference) SQL query by executing both
    on the specified SQLite database and comparing their results.

    Parameters:
    - evaluation_entry (SQLEvaluationEntry): The evaluation entry to evaluate.
    - conn (sqlite3.Connection): The SQLite connection to use. If None, a new connection will be created.
    - close_conn (bool): Whether to close the connection after evaluation.
    Returns:
    - SQLEvaluationEntry: The evaluation entry containing the results of the evaluation.
    """
    print("----------RECAP----------")
    print("Question: " + evaluation_entry.question)
    print("Gold SQL: " + evaluation_entry.gold_sql)
    print("Generated SQL: " + evaluation_entry.generated_sql)

    try:
        isCorrect = evaluateSQL(evaluation_entry.generated_sql, evaluation_entry.gold_sql, evaluation_entry.db_path)

        evaluation_entry.isCorrect = isCorrect
    except Exception as e:
        print(f"An error occurred: {e}")
        evaluation_entry.isCorrect = False
        evaluation_entry.generated_sql = evaluation_entry.generated_sql + "\n" + str(e)
    return evaluation_entry

def evaluateModel(model_name: str, dataset="spider", split="train", promptTemplate=config["prompt_template"]):
    print("Evaluating model: " + model_name)
    result = []
    if dataset == "spider":
        spider_instances = load_spider(split)
        for instance in tqdm(spider_instances):
            evaluation_entry = generateSQLEvaluationEntry(model_name, instance, split=split, promptTemplate=promptTemplate)
            result.append(evaluateSQLGenerationEntry(evaluation_entry))
    # Convert result to pandas dataframe
    df = objects_to_dataframe(result)
    return df

def evaluateFromFile(filePath: str):
    result = []
    sqlEvalEntries = loadToObjectsFromFile(filePath, SQLEvaluationEntry, file_type="csv")
    for sqlEvalEntry in tqdm(sqlEvalEntries):
        result.append(evaluateSQLGenerationEntry(sqlEvalEntry))
    # Convert result to pandas dataframe
    df = objects_to_dataframe(result)
    return df
