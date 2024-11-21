import os

import ollama
from openai import OpenAI

from core.data_loader import get_spider_db_path
from core.sql_tools import getDatabaseSchemaForPrompt
from core.utils import config, cleanLLMResponse
from models.SQLEvaluationEntry import SQLEvaluationEntry
from models.SpiderDataset import SpiderDataset



def prompt(model_name, promptTemplate=config["prompt_template"], **kwargs):
    apiModels = ['gpt-4o-mini', 'gpt-4o']
    if model_name in apiModels:
        print("Using: " + model_name)
        client = OpenAI(api_key=None)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": promptTemplate.format(**kwargs)
                }
            ], model=model_name
        )
        return response.choices[0].message.content


    return ollama.generate(model=model_name, prompt=promptTemplate.format(**kwargs))['response']

# @suppress_prints
def generateSQL(model_name, promptTemplate=config["prompt_template"], db_path="", **kwargs):
    # kwargs should include arguments for the prompt template
    kwargs["db_path"] = db_path
    print("--" * 50)
    print(promptTemplate.format(**kwargs))
    print("--" * 50)

    return prompt(model_name, promptTemplate=promptTemplate, **kwargs)

# @suppress_prints
def generateSQLEvaluationEntry(model_name: str, spider_dataset_entry: SpiderDataset):
    request = spider_dataset_entry.question
    schema = getDatabaseSchemaForPrompt(get_spider_db_path(spider_dataset_entry.db_id))

    response = generateSQL(model_name=model_name,
                                db_path=get_spider_db_path(spider_dataset_entry.db_id),
                                request=request,
                                schema=schema)
    print("----------RESPONSE----------")
    print(response)
    generated_sql = cleanLLMResponse(response)

    return SQLEvaluationEntry(
        get_spider_db_path(spider_dataset_entry.db_id),
        generated_sql,
        spider_dataset_entry.query,
        spider_dataset_entry.question
    )

