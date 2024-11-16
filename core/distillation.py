from core.generation import prompt
from models.DistillationEntry import DistillationEntry


def generateDistillationEntry(model_name: str, promptTemplate: str, question: str, gold_solution: str, schema: str):
    distillationEntry = DistillationEntry(
        teacher_model_name=model_name,
        question=question,
        schema=schema,
        gold_solution=gold_solution,
        reasoning="",
        verification_solution="",
        isVerified=None
    )
    model_response = prompt(
        model_name,
        promptTemplate=promptTemplate,
        problem=question,
        solution=gold_solution,
        schema=schema
    )
