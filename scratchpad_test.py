from main import load_spider, generateSQLEvaluationEntry, evaluateSQLGenerationEntry, config
from sql_tools import inspect_database, formatSchemaForPrompt


def test():
    train_data = load_spider("train")
    generated_sql = "SELECT count(*) FROM head WHERE age  >  53"
    sqlEvaluationEntry = generateSQLEvaluationEntry(generated_sql, train_data[0])
    evaluateSQLGenerationEntry(sqlEvaluationEntry)

def testSchemaRetrieval():
    db_id = "department_management"
    sample_db_path = config["spider_root_path"] + "/database/" + db_id + "/" + db_id + ".sqlite"
    db_info = inspect_database(db_path=sample_db_path, include_sample_data=True)

    print(db_info)
    print("_" * 20)

    # print_schema(db_info)
    # print("_" * 20)

    print(formatSchemaForPrompt(db_info))

# test()
print(config['prompt_template'])



