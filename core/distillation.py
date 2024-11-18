from typing import Optional

import pandas
from tqdm import tqdm

from core.data_loader import load_spider, get_spider_db_path
from core.evaluation import evaluateSQLGenerationEntry
from core.generation import prompt
from core.sql_tools import getDatabaseSchemaForPrompt
from core.utils import cleanLLMResponse, config, objects_to_dataframe, loadToObjectsFromFile
from models.DistillationEntry import DistillationEntry
from models.SQLEvaluationEntry import SQLEvaluationEntry
from pathlib import Path

def generateDistillationEntry(
        modelName: str,
        question: str,
        goldSolution: str,
        schema: str,
        promptTemplate: str = config["knowledge_distillation_generation_template"],
):

    model_response = prompt(
        modelName,
        promptTemplate=promptTemplate,
        problem=question,
        solution=goldSolution,
        schema=schema
    )

    reasoning = cleanLLMResponse(model_response, openTag="<reasoning>", closeTag="</reasoning>")
    distillationEntry = DistillationEntry(
        teacher_model_name=modelName,
        question=question,
        schema=schema,
        gold_solution=goldSolution,
        reasoning=reasoning,
        verification_solution="",
        isVerified=None
    )

    return distillationEntry


#Load distillation entries from file
def verifyDistillationFromFile(
        db_path: str,
        verifierModelName: str,
        distillationFilePath: str,
        promptTemplate: str = config["knowledge_distillation_verification_template"],
) -> list[DistillationEntry]:
    distillationEntries = loadToObjectsFromFile(distillationFilePath, DistillationEntry)
    verificationResult = []
    for distillationEntry in tqdm(distillationEntries):
        distillationEntry = verifyDistillationEntry(
            distillationEntry=distillationEntry,
            model_name=verifierModelName,
            db_path=db_path,
            promptTemplate=promptTemplate
        )
        verificationResult.append(distillationEntry)

    return verificationResult




def verifyDistillationEntry(
        distillationEntry: DistillationEntry,
        model_name: str,
        db_path: str,
        promptTemplate: str = config["knowledge_distillation_verification_template"]) -> DistillationEntry:
    '''
    This function verifies the logical reasoning of the generated distillation by prompting another model
    given the question and the reasoning to generate sql solution which would then be evaluated.
    :param distillationEntry:
    :param model_name:
    :param db_path:
    :param promptTemplate:
    :return:
    '''
    model_response = prompt(
        model_name,
        promptTemplate=promptTemplate,
        problem=distillationEntry.question,
        schema=distillationEntry.schema,
        reasoning=distillationEntry.reasoning
    )

    distillationEntry.verification_solution = cleanLLMResponse(model_response, openTag="<final answer>", closeTag="</final answer>")
    sqlEvaluationEntry = SQLEvaluationEntry(
        db_path=db_path,
        generated_sql=distillationEntry.verification_solution,
        gold_sql=distillationEntry.gold_solution,
        question=distillationEntry.question
    )
    sqlEvaluationEntry = evaluateSQLGenerationEntry(sqlEvaluationEntry)

    distillationEntry.isVerified = sqlEvaluationEntry.isCorrect

    return distillationEntry

def distillKnowledge(
        teacher_model_name: str,
        student_model_name: Optional[str] = None,
        dataset="spider",
        split="train",
        useCache=False,
        batchRange: Optional[tuple[int, int]] = None
):
    cachePath = Path(f'{config["cache_path"]}/distillation/{teacher_model_name}_{dataset}_{split}.csv')
    if useCache and cachePath.exists():
        return pandas.read_csv(cachePath)

    result = []
    if dataset == "spider":
        spider_instances = load_spider(split, batchRange=batchRange)
        for instance in tqdm(spider_instances):
            db_path = get_spider_db_path(instance.db_id)
            schema = getDatabaseSchemaForPrompt(db_path)
            distillationEntry = generateDistillationEntry(
                modelName=teacher_model_name,
                question=instance.question,
                goldSolution=instance.query,
                schema=schema,
            )
            student_model_name = teacher_model_name if student_model_name is None else student_model_name
            distillationEntry = verifyDistillationEntry(
                distillationEntry=distillationEntry,
                model_name=student_model_name,
                db_path=db_path
            )
            result.append(distillationEntry)


    pd = objects_to_dataframe(result)
    if useCache:
        pd.to_csv(cachePath)

    return pd

