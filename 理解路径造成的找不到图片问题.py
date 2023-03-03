import sys
from pathlib import Path

# 以项目根目录为基准，什么文件都好找了, 不然总是找不到文件。这是从 yolov5 官方代码里学到的！！！
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 当前文件若在项目根目录下则填 0，若在下一级目录里则填 1，以此类推……
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

print(ROOT)  # D:\code\deep_learning\projects\yolov5-coco-dataset
