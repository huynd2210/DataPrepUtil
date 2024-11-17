from core.distillation import distillKnowledge
from core.evaluation import evaluateModel
from core.sql_tools import getDatabaseSchemaForPrompt

import sqlite3
import pandas as pd
import ollama

from models.SpiderDataset import SpiderDataset
from models.SQLEvaluationEntry import SQLEvaluationEntry
from core.utils import suppress_prints, objects_to_dataframe, load_json_to_class, config, cleanLLMResponse, \
    compare_dataframes, print_comparison_results, isDataFrameEqual

from core.data_loader import load_spider, get_spider_db_path




def analyseEvaluation(evaluationResultDf: pd.DataFrame):
    counter = 0
    for index, row in evaluationResultDf.iterrows():
        if row["isCorrect"]:
            counter += 1

    print("Accuracy: " + str(counter / len(evaluationResultDf)))

if __name__ == '__main__':
    # model_name = "qwen2.5-coder:latest"

    model_name = "llama3.1:8b-instruct-q4_0"
    datasetName = "spider"
    result = evaluateModel(model_name, datasetName)
    analyseEvaluation(result)
    print("----RESULT----")
    print(result)
    outputName = f"{model_name.replace(':', '-')}_{datasetName}_result.csv"
    print(f"Output saved to {outputName}")
    result.to_csv(outputName)

    # data = distillKnowledge(model_name, dataset=datasetName)
    # outputName = f"{model_name.replace(':', '-')}_distilled_data.csv"
    # print(f"Output saved to {outputName}")
    # data.to_csv(outputName)

    # df = pd.read_csv("qwen2.5-coder-7b-instruct_distilled_data_spider.csv")
    # print(df["reasoning"].iloc[0])


    # generatedSQL = """
    # SELECT COUNT(*)
    # FROM head AS h
    # WHERE h.age > 56;
    # """
    # gold = """SELECT COUNT(*) FROM head WHERE age > 56;"""
    # path = "spider_data/database/department_management/department_management.sqlite"
    # sample = SQLEvaluationEntry(path, generatedSQL, gold, "question")
    # evaluateSQLGenerationEntry(sample)