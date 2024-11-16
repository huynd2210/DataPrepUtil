from dataclasses import dataclass

@dataclass
class AlpacaFormat:
    instruction: any = None
    input: any = None
    output: any = None
    system: any = None
    history: any = None
