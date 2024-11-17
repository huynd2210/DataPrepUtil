import sqlite3

import pandas as pd
from tqdm import tqdm

from core.data_loader import load_spider
from core.generation import generateSQLEvaluationEntry
from core.utils import isDataFrameEqual, objects_to_dataframe
from models.SQLEvaluationEntry import SQLEvaluationEntry


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
        # Connect to the database
        if conn is None:
            conn = sqlite3.connect(evaluation_entry.db_path)

        # Execute both queries and fetch results as dataframes
        generated_df = pd.read_sql_query(evaluation_entry.generated_sql, conn)
        gold_df = pd.read_sql_query(evaluation_entry.gold_sql, conn)

        if close_conn:
            conn.close()

        # Check if both dataframes are identical
        # isIdentical, comparisonResult = compare_dataframes(generated_df, gold_df)
        # print_comparison_results((isIdentical, comparisonResult))

        isCorrect = isDataFrameEqual(generated_df, gold_df)

        if isCorrect:
            print("The generated SQL matches the gold SQL.")
            evaluation_entry.isCorrect = True
            return evaluation_entry
        else:
            print("--------------------")
            print("The generated SQL does not match the gold SQL.")

            # Show differences
            diff = pd.concat([generated_df, gold_df]).drop_duplicates(keep=False)
            print("Differences:")
            print(diff)

            evaluation_entry.isCorrect = False
            return evaluation_entry

    except Exception as e:
        print(f"An error occurred: {e}")
        evaluation_entry.isCorrect = False
        evaluation_entry.generated_sql = evaluation_entry.generated_sql + "\n" + str(e)
    return evaluation_entry

def evaluateModel(model_name: str, dataset="spider", split="train"):
    result = []
    if dataset == "spider":
        spider_instances = load_spider(split)
        for instance in tqdm(spider_instances):
            evaluation_entry = generateSQLEvaluationEntry(model_name, instance)
            result.append(evaluateSQLGenerationEntry(evaluation_entry))
    # Convert result to pandas dataframe
    pd = objects_to_dataframe(result)
    return pd