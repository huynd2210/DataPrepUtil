import pprint
import textwrap
from typing import Optional

import pandas as pd

from DatasetFormat.AlpacaFormat import AlpacaFormat
from core.data_handler import distillationToAlpaca, convertDistillationEntriesToAlpaca
from core.distillation import distillKnowledge, distillUnverifiedEntries, redistillEntries, generateVanillaEntry
from core.evaluation import evaluateModel
from core.utils import merge_csv_files, load_json_to_class, loadToObjectsFromFile, config
from models.DistillationEntry import DistillationEntry


def prettyPrintCSV(path: str, chosenColumns: Optional[list[str]] = None):
    df = pd.read_csv(path, on_bad_lines='warn')
    if chosenColumns is not None:
        df = df[chosenColumns]
    textWrapWidth = 160
    batch_size = 10  # Number of rows to display at a time

    # Iterate in batches
    for start_idx in range(0, len(df), batch_size):
        end_idx = start_idx + batch_size
        batch = df.iloc[start_idx:end_idx]

        for index, row in batch.iterrows():
            if row["isVerified"] == 0 or row["isVerified"] is None:

                print(f" Question: {row['question']}" , end="\n\n")
                print(textwrap.fill("reasoning: " + row["reasoning"], textWrapWidth), end="\n\n")
                if type(row["verification_solution"]) == float:
                    print("0" * 20)
                    print(row["verification_solution"])
                print(textwrap.fill("gold_solution: " + row["gold_solution"], textWrapWidth), end="\n\n")
                print(textwrap.fill("verification_solution: " + str(row["verification_solution"]), textWrapWidth), end="\n\n")
                print(textwrap.fill("isVerified: " + str(row["isVerified"]), textWrapWidth), end="\n\n")
                # print(textwrap.fill("schema: " + row["schema"], textWrapWidth), end="\n\n")
                print("-" * 20)
                print("\n")

        # Pause before displaying the next batch
        if end_idx < len(df):  # Avoid asking for input after the last batch
            input("Press Enter to view the next set of rows...")
    # pprint.pprint(df)


def analyseEvaluation(evaluationResultDf: pd.DataFrame):
    counter = 0
    for index, row in evaluationResultDf.iterrows():
        if row["isCorrect"]:
            counter += 1

    print("Accuracy: " + str(counter / len(evaluationResultDf)))

def distillWrapper(
        model_name: str,
        student_model_name: str = None,
        dataset="spider",
        split="train",
        batchRange: Optional[tuple[int, int]] = None,
        isBatchMode=False
):
    data = distillKnowledge(model_name, student_model_name, dataset=dataset, split=split, batchRange=batchRange, isBatchMode=isBatchMode)
    if batchRange is not None:
        outputName = f"{model_name.replace(':', '-')}_distilled_data_{dataset}_{split}_{batchRange[0]}_{batchRange[1]}.csv"
    else:
        outputName = f"{model_name.replace(':', '-')}_distilled_data_{dataset}_{split}.csv"
    print(f"Output saved to {outputName}")
    data.to_csv(outputName)

def generateVanillaData(dataset, split):
    data = generateVanillaEntry(dataset=dataset, split=split)
    outputName = f"vanilla_data_{dataset}_{split}.csv"
    print(f"Output saved to {outputName}")
    data.to_csv(outputName)


def redistillWrapper(
        file_path: str,
        model_name: str,
):
    pd = distillUnverifiedEntries(file_path, model_name)
    outputName = file_path.replace(".csv", "_redistilled.csv")
    print(f"Output saved to {outputName}")
    pd.to_csv(outputName)

if __name__ == '__main__':
    # generateVanillaData(dataset="spider", split="train")


    # redistillEntries(
    #     inputFilePath="datasets/distilled_spider_train/spider-train-distilled.csv",
    #     outputFilePath="datasets/distilled_spider_train/spider-train-redistilled.csv",
    #     model_name="gpt-4o",
    #     entriesIndex=entriesIndex
    # )

    # model_name = "gpt-4o-mini"
    # # student_model_name = "gpt-4o-mini"
    # redistillWrapper(
    #     "datasets/distilled_spider_train/gpt-4o-mini-distilled-spider-train.csv",
    #     "gpt-4o",
    #     "gpt-4o-mini",
    # )

    # student_model_name = "qwen2.5-coder:7b-instruct"

    batchRange = (3301, 3400)
    # batchRange = (2601, 2700)
    distillWrapper(model_name="gpt-4o", dataset="bird", split="train", batchRange=batchRange)


    # convertDistillationEntriesToAlpaca(
    #     inputFilePath="vanilla_data_spider_train.csv",
    #     outputFilePath="vanilla_data_spider_train-alpaca.csv",
    #     outputExtension="json"
    # )


    # inputFilePath = "datasets/distilled_spider_train/spider-train-redistilled-alpaca.csv"
    # distillationEntries = loadToObjectsFromFile(inputFilePath, AlpacaFormat)
    # print(distillationEntries[0])


    # model_name = "llama3.1:8b-instruct-q4_0"
    #
    # model_name = "NyanDoggo/Qwen2.5-Coder-3B-Instruct-Spider-Reasoning"
    # split="test"
    # datasetName = "spider"
    # result = evaluateModel(model_name, datasetName, split=split, promptTemplate=config["alpaca_inference_template"])
    # analyseEvaluation(result)
    # print("----RESULT----")
    # print(result)
    # model_name = model_name.replace("/", "-")
    # outputName = f"{model_name.replace(':', '-')}_{datasetName}_result.csv"
    # print(f"Output saved to {outputName}")
    # result.to_csv(outputName)

    # generatedSQL = """
    # SELECT COUNT(*)
    # FROM head AS h
    # WHERE h.age > 56;
    # """
    # gold = """SELECT COUNT(*) FROM head WHERE age > 56;"""
    # path = "spider_data/database/department_management/department_management.sqlite"
    # sample = SQLEvaluationEntry(path, generatedSQL, gold, "question")
    # evaluateSQLGenerationEntry(sample)
