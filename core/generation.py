import os

import instructor
import ollama
from openai import OpenAI

from core.data_loader import get_spider_db_path
from core.prompt_delivery import Prompt
from core.sql_tools import getDatabaseSchemaForPrompt
from core.utils import config, cleanLLMResponse
from models.SQLEvaluationEntry import SQLEvaluationEntry
from models.SQLQuery import SQLQuery
from models.SQLDataset import SQLDataset


def askAI(model, db_path, request, promptTemplate=config["alpaca_inference_template"]):
    schema = getDatabaseSchemaForPrompt(db_path)
    prompt = Prompt(
        modelName=model,
        promptTemplate=promptTemplate,
        request=request,
        schema=schema
    )
    return prompt.deliver()
# @suppress_prints
def generateSQL(model_name, promptTemplate=config["prompt_template"], db_path="", **kwargs):
    # kwargs should include arguments for the prompt template
    kwargs["db_path"] = db_path
    print("--" * 50)
    print(promptTemplate.format(**kwargs))
    print("--" * 50)
    prompt = Prompt(
        modelName=model_name,
        promptTemplate=promptTemplate,
        **kwargs
    )
    return prompt.deliver()

# @suppress_prints
def generateSQLEvaluationEntry(
        model_name: str,
        spider_dataset_entry: SQLDataset,
        isInstructor=False,
        split="train",
        promptTemplate=config["prompt_template"]
):
    request = spider_dataset_entry.question
    schema = getDatabaseSchemaForPrompt(get_spider_db_path(spider_dataset_entry.db_id, split=split))

    response = generateSQL(model_name=model_name,
                                promptTemplate=promptTemplate, #promptTemplate
                                db_path=get_spider_db_path(spider_dataset_entry.db_id, split=split),
                                request=request,
                                schema=schema)
    print("----------RESPONSE----------")
    print(response)

    if isInstructor:
        generated_sql = response.sql_query
    else:
        generated_sql = cleanLLMResponse(response)

    return SQLEvaluationEntry(
        db_path=get_spider_db_path(spider_dataset_entry.db_id, split=split),
        generated_sql=generated_sql,
        gold_sql=spider_dataset_entry.query,
        question=spider_dataset_entry.question,
        response=response
    )

