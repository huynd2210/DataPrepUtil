import ollama

from core.evaluation import evaluateSQLGenerationEntry
from core.generation import generateSQLEvaluationEntry
from main import load_spider, config
from core.sql_tools import retrieveDatabaseSchema, formatSchemaForPrompt


def test():
    train_data = load_spider("train")
    generated_sql = "SELECT count(*) FROM head WHERE age  >  53"
    sqlEvaluationEntry = generateSQLEvaluationEntry(generated_sql, train_data[0])
    evaluateSQLGenerationEntry(sqlEvaluationEntry)

def testSchemaRetrieval():
    db_id = "department_management"
    sample_db_path = config["spider_root_path"] + "/database/" + db_id + "/" + db_id + ".sqlite"
    db_info = retrieveDatabaseSchema(db_path=sample_db_path, include_sample_data=True)

    print(db_info)
    print("_" * 20)

    # print_schema(db_info)
    # print("_" * 20)

    print(formatSchemaForPrompt(db_info))

# testSchemaRetrieval()
# load_spider("train")
# test()
# print(config['prompt_template'])
# print(ollama.generate(model="llama3.1:latest", prompt="what is the answer to life, the universe, and everything?")['response'])



