from pydantic import BaseModel


class SQLReasoning(BaseModel):
    reasoning: str
    sql_query: str