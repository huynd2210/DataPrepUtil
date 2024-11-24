from typing import Optional

from core.utils import load_json_to_class, suppress_prints, config
from models.SpiderDataset import SpiderDataset


@suppress_prints
def load_spider(split: str = "train", batchRange: Optional[tuple[int, int]] = None) -> list[SpiderDataset]:
    batchResult = []
    splitFilePath = {
        "train": 'old/train_spider_clean.json',
        "dev": 'old/dev_spider_clean.json',
        "test": 'old/test_spider_clean.json',
        "others": 'old/train_spider_others_clean.json'
    }
    jsonFilePath = splitFilePath[split]
    instances = load_json_to_class(jsonFilePath, SpiderDataset)
    print("Loading batch range: " + str(batchRange))
    for i in range(len(instances)):
        instance = instances[i]
        if batchRange is not None and i in range(batchRange[0], batchRange[1] + 1):
            batchResult.append(instance)
        else:
            print("Skipping instance " + str(i))
        # print(instance)

    if batchRange is not None:
        instances = batchResult
    return instances

def get_spider_db_path(db_id, spiderRootPath=config["spider_root_path"], split="train"):
    if split == "train" or split == "others":
        return f"{spiderRootPath}/database/{db_id}/{db_id}.sqlite"
    else:
        return f"{spiderRootPath}/test_database/{db_id}/{db_id}.sqlite"

