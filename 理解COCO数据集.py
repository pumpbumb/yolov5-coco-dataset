import json
import sys
from pathlib import Path

# -------------------------------------------------------------- #
# 找得到路径就不用加下面一段，找不到再加吧
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# -------------------------------------------------------------- #

json_path = "./coco_dataset/test_annotations/image_info_test2017.json"
json_labels = json.load(open(json_path, "r"))

print(json_labels["info"])
