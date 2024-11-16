
class SpiderDatasetStandardClass:
    def __init__(self, **kwargs):
        self.db_id = kwargs.get('db_id', None)
        self.query = kwargs.get('query', None)
        self.question = kwargs.get('question', None)

    def __str__(self):
        return f"db_id: {self.db_id}, query: {self.query}, question: {self.question}"

    def __repr__(self):
        return f"db_id: {self.db_id}, query: {self.query}, question: {self.question}"

