from core.utils import load_json_to_class, suppress_prints
from main import config
from models.SpiderDataset import SpiderDataset


@suppress_prints
def load_spider(split: str = "train") -> list[SpiderDataset]:
    if split == "train":
        instances = load_json_to_class('old/train_spider_clean.json', SpiderDataset)
        for instance in instances:
            print(instance)
        return instances

def get_spider_db_path(db_id, spiderRootPath=config["spider_root_path"], split="train"):
    if split == "train":
        return f"{spiderRootPath}/database/{db_id}/{db_id}.sqlite"
    else:
        return f"{spiderRootPath}/test_database/{db_id}/{db_id}.sqlite"

