from core.utils import save_list_to_jsonl


class Request:
    def __init__(self, custom_id, method, url, body):
        self.custom_id = custom_id
        self.method = method
        self.url = url
        self.body = body

    def __dict__(self):
        return {"custom_id": self.custom_id, "method": self.method, "url": self.url, "body": self.body.__dict__()}


class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __dict__(self):
        return {"role": self.role, "content": self.content}


class Body:
    def __init__(self, model, messages, max_tokens):
        self.model = model
        self.messages = [message.__dict__() for message in messages]
        self.max_tokens = max_tokens

    def __dict__(self):
        return {"model": self.model, "messages": self.messages, "max_tokens": self.max_tokens}

class BatchRequestController:
    def __init__(self):
        self.requests = []

    def _create_jsonl_requests(self, requests, fileName = "batch_requests.jsonl"):
        save_list_to_jsonl(requests, fileName)

    def upload(self, requests):
        fileName = "batch_requests.jsonl"
        self._create_jsonl_requests(requests, fileName=fileName)
        from openai import OpenAI
        client = OpenAI()

        batch_input_file = client.files.create(
            file=open("batchinput.jsonl", "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "nightly eval job"
            }
        )
        print("Batch request uploaded")
        print(batch)
        save_list_to_jsonl([batch], "batch.jsonl")