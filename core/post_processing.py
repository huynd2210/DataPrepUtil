import pandas as pd
from tqdm import tqdm

from core.evaluation import evaluateSQL
from core.prompt_delivery import Prompt
from core.utils import cleanLLMResponse

def reevaluateCSVResult(file_path, targetColumn, retrievalModel):
    df = pd.read_csv(file_path)
    isCorrectColumn = "isCorrect"
    for index, row in tqdm(df.iterrows()):
        if not row[isCorrectColumn]:
            value = row[targetColumn]
            print("Reevaluating: ", value)
            response = retrieveSQLFromResponse(retrievalModel, value)
            response = cleanLLMResponse(response)
            df.at[index, targetColumn] = response
            print("Reevaluated: ", df.loc[index, targetColumn])
            isCorrect = evaluateSQL(response, row['gold_sql'], row['db_path'])
            print("isCorrect: ", isCorrect)
            df.at[index, isCorrectColumn] = isCorrect
            # if not isCorrect:
            #     ic(row, response)


    modified_file_path = file_path.replace(".csv", "_reevaluated.csv")
    df.to_csv(modified_file_path, index=False)


def retrieveSQLFromResponse(retrievalModel, response):
    promptTemplate = """
    Retrieve the sql in the text below. Response only with the sql and nothing else.
    {response}

    """
    prompt = Prompt(
        modelName=retrievalModel,
        promptTemplate=promptTemplate,
        response=response
    )

    return prompt.deliver()