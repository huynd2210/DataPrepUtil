from DatasetFormat.AlpacaFormat import AlpacaFormat
from core.utils import config
from models.DistillationEntry import DistillationEntry


def distillationToAlpaca(
        distillationEntries: list[DistillationEntry],
        alpaca_instruction_template: str = config["alpaca_instruction_template"]
):
    alpaca = []
    for distillationEntry in distillationEntries:
        reasoning_with_answer = distillationEntry.reasoning + "\n<final answer>" + distillationEntry.gold_solution + "</final answer>"
        instruction = alpaca_instruction_template.format(request=distillationEntry.question, schema=distillationEntry.schema)
        alpaca.append(
            AlpacaFormat(instruction=instruction, output=reasoning_with_answer)
        )
    return alpaca
