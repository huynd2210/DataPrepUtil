import instructor
import ollama
from models.SQLQuery import SQLQuery
from openai import OpenAI

from core.utils import config
from transformers import AutoModelForCausalLM, AutoTokenizer

class TransformerCache:
    transformerModel = None
    tokenizer = None

    @classmethod
    def get_or_create(cls, model_name):
        if cls.transformerModel is None:
            cls.transformerModel = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls.transformerModel, cls.tokenizer


#Delivers prompt to LLM over different interfaces, currently prototype, enough for the time being.
#In the future add other interfaces and ways to register model -> prompt strategy

class Prompt:
    """
    promptTemplate: str -> The message content
    """
    def __init__(
            self,
            modelName: str,
            promptTemplate: str = config["prompt_template"],
            isInstructor: bool = False,
            **kwargs
    ):
        self.modelName = modelName
        self.messageContent = promptTemplate.format(**kwargs)
        self.isInstructor = isInstructor
        self.defaultPromptStrategy = self._deliverOllamaPrompt
        self.modelPromptStrategyMap = {
            "gpt-4o": self._deliverAPIPrompt,
            "gpt-4o-mini": self._deliverAPIPrompt,
            "NyanDoggo/Qwen2.5-Coder-3B-Instruct-Spider-Vanilla": self._deliverTransformersTokenizerPrompt,
        }

        self.transformerModel = None
        self.tokenizer = None

    def deliver(self):
        return self._setupPromptStrategy()()

    def _setupPromptStrategy(self):
        if self.isInstructor:
            return self._deliverPromptInstructor
        return self.modelPromptStrategyMap.get(self.modelName, self.defaultPromptStrategy)

    def _setupClient(self):
        if self.isInstructor:
            return self._setupInstructorClient()
        return OpenAI(api_key=None)

    def _setupInstructorClient(self):
        client = instructor.from_openai(
            OpenAI(
                base_url=config["default_ollama_server"],
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )
        return client

    def _deliverPromptInstructor(self, structuredOutputClass=SQLQuery):
        client = self._setupClient()
        response = client.chat.completions.create(
            model=self.modelName,
            messages=[
                {
                    "role": "user",
                    "content": self.messageContent,
                }
            ],
            response_model=structuredOutputClass,
        )
        return response

    def _deliverAPIPrompt(self):
        client = self._setupClient()
        return client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": self.messageContent
                }
            ], model=self.modelName,
            temperature=0.1
        ).choices[0].message.content

    def _deliverOllamaPrompt(self):
        return ollama.generate(model=self.modelName, prompt=self.messageContent)['response']

    def _deliverTransformersTokenizerPrompt(self):
        device = "cuda"  # the device to load the model onto

        if self.transformerModel is None:
            self.transformerModel, self.tokenizer = TransformerCache.get_or_create(self.modelName)

        messages = [{"role": "user", "content": self.messageContent}]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.transformerModel.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        return response