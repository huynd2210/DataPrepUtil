from typing import Optional

from core.data_loader import load_spider, get_spider_db_path
from core.evaluation import evaluateSQLGenerationEntry
from core.generation import prompt
from core.sql_tools import getDatabaseSchemaForPrompt
from core.utils import cleanLLMResponse, config, objects_to_dataframe
from models.DistillationEntry import DistillationEntry
from models.SQLEvaluationEntry import SQLEvaluationEntry


def generateDistillationEntry(
        model_name: str,
        question: str,
        gold_solution: str,
        schema: str,
        promptTemplate: str = config["knowledge_distillation_generation_template"]
):
    model_response = prompt(
        model_name,
        promptTemplate=promptTemplate,
        problem=question,
        solution=gold_solution,
        schema=schema
    )

    reasoning = cleanLLMResponse(model_response, openTag="<reasoning>", closeTag="</reasoning>")
    distillationEntry = DistillationEntry(
        teacher_model_name=model_name,
        question=question,
        schema=schema,
        gold_solution=gold_solution,
        reasoning=reasoning,
        verification_solution="",
        isVerified=None
    )

    return distillationEntry

def verifyDistillationEntry(
        distillationEntry: DistillationEntry,
        model_name: str,
        db_path: str,
        promptTemplate: str = config["knowledge_distillation_verification_template"]) -> DistillationEntry:
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

    if sqlEvaluationEntry.isCorrect:
        distillationEntry.isVerified = True
    else:
        distillationEntry.isVerified = False

    return distillationEntry

def distillKnowledge(teacher_model_name: str, student_model_name: Optional[str] = None, dataset="spider", split="train"):
    result = []
    if dataset == "spider":
        spider_instances = load_spider(split)
        for instance in spider_instances:
            db_path = get_spider_db_path(instance.db_id)
            schema = getDatabaseSchemaForPrompt(db_path)
            distillationEntry = generateDistillationEntry(
                model_name=teacher_model_name,
                question=instance.question,
                gold_solution=instance.query,
                schema=schema
            )
            student_model_name = teacher_model_name if student_model_name is None else student_model_name
            distillationEntry = verifyDistillationEntry(
                distillationEntry=distillationEntry,
                model_name=student_model_name,
                db_path=db_path
            )
            result.append(distillationEntry)
            break

    pd = objects_to_dataframe(result)
    return pd

