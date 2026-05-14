# 基于 YOLO 的织物疵点检测实验报告

## 一、实验目的

本实验采用机器视觉方法对织物图像进行质量检测，目标是在输入织物图像中自动识别并定位破洞、水渍、三丝、结头、断经、断氨纶等疵点。织物疵点通常具有面积小、形状不规则、类别相似度高、背景纹理重复等特点，人工检测不仅效率低，而且容易受疲劳和主观经验影响。因此，本实验使用目标检测模型完成疵点的自动定位和分类，为织物质量检测提供可复现的算法流程。

## 二、数据集与标签处理

原始训练数据位于 `yolo_8_2`，由 `smartdiagnosisofclothflaw_round1train1_datasets_partA/partA` 中的图像和 `Annotations/anno_train.json` 转换得到。原始标注格式为：

```json
{
  "name": "xxx.jpg",
  "defect_name": "结头",
  "bbox": [x1, y1, x2, y2]
}
```

YOLO 训练要求每张图片对应一个 `.txt` 标签文件，单行格式为：

```text
class_id x_center y_center width height
```

其中坐标均需归一化到 `[0, 1]`，`class_id` 使用 README 中的 category id。正常无疵点图片保留为空 `.txt` 文件，作为负样本参与训练。

本实验的数据划分结果如下：

| 数据集 | 图片数 | 标签文件数 | 说明 |
| --- | ---: | ---: | --- |
| train | 2924 | 2924 | 训练集，占 80% |
| val | 732 | 732 | 验证集，占 20% |
| 合计 | 3656 | 3656 | 其中 2387 张含疵点，1269 张为正常图 |

标签框总数为 3916 个，类别分布如下：

| 类别 id | 数量 | 类别 id | 数量 | 类别 id | 数量 | 类别 id | 数量 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 149 | 2 | 217 | 3 | 528 | 4 | 782 |
| 5 | 56 | 6 | 57 | 7 | 89 | 8 | 99 |
| 9 | 162 | 10 | 119 | 11 | 71 | 12 | 332 |
| 13 | 147 | 14 | 100 | 15 | 141 | 16 | 76 |
| 17 | 297 | 18 | 168 | 19 | 181 | 20 | 145 |

数据集配置文件为：

```yaml
path: D:/Study/rh/workspace/machineVisionWork-0511/data/smartdiagnosisofclothflaw_round1train1_datasets_partA/partA/yolo_8_2
train: images/train
val: images/val

nc: 21
names:
  0: normal
  1: hole
  2: stain
  3: three_threads
  4: knot
  5: skip_pattern
  6: centipede
  7: fuzz_ball
  8: thick_warp
  9: loose_warp
  10: broken_warp
  11: hanging_warp
  12: thick_weft
  13: weft_shrinkage
  14: size_spot
  15: warping_knot
  16: skip
  17: broken_spandex
  18: density_or_color_block
  19: mark
  20: misc_weave_defect
```

> 注：当前 partA 文件夹中实际存在 2387 张缺陷图和 1269 张正常图。`anno_train.json` 中还有部分标注图片名在当前文件夹中未找到，因此转换时只使用了本地实际存在的图片。

## 三、检测方法选择与原因

本实验选择 YOLO 系列单阶段目标检测算法进行织物疵点检测。该方法把目标定位和类别识别统一为一个端到端网络任务，能够直接输出疵点的类别、位置框和置信度。

选择 YOLO 的原因如下：

1. 检测速度快。YOLO 属于单阶段检测方法，不需要先生成候选框再分类，适合工业视觉中实时或准实时检测场景。
2. 能同时完成定位与分类。织物质量检测不仅要判断是否有缺陷，还需要指出缺陷位置，YOLO 的检测框输出正好满足需求。
3. 对小目标和多类别检测有较成熟的训练流程。织物疵点常常面积较小、数量不均衡，YOLO 的多尺度特征融合和数据增强策略能提升模型对不同尺度缺陷的适应能力。
4. 工程实现方便。数据格式、训练、验证、可视化和模型导出流程完整，便于完成课程实验和后续应用。

传统图像处理方法如阈值分割、边缘检测、形态学处理等对背景单一、缺陷明显的图像有效，但织物图像纹理复杂，疵点形态变化大，固定阈值和人工特征难以覆盖所有类别。因此，本实验采用深度学习目标检测方法作为主要方案。

## 四、具体实施步骤

### 4.1 数据格式转换

首先读取 `Annotations/anno_train.json`，按照图片名将多个标注框归并到同一张图片下。然后将 `xyxy` 格式边界框转换为 YOLO 所需的归一化中心点格式：

```text
x_center = ((x1 + x2) / 2) / image_width
y_center = ((y1 + y2) / 2) / image_height
width    = (x2 - x1) / image_width
height   = (y2 - y1) / image_height
```

输出目录采用 YOLO 常用结构：

```text
yolo_8_2/
  images/train
  images/val
  labels/train
  labels/val
  dataset.yaml
```

### 4.2 数据增强与训练

训练阶段使用预训练权重初始化模型，再在织物疵点数据上微调。训练参数如下：

| 参数 | 值 |
| --- | --- |
| model | `yolo26n.pt` |
| epochs | 100 |
| batch | 88 |
| imgsz | 640 |
| optimizer | SGD |
| initial lr | 0.01 |
| momentum | 0.937 |
| weight decay | 0.0005 |
| mosaic | 1.0 |
| fliplr | 0.5 |
| hsv_s / hsv_v | 0.7 / 0.4 |
| close_mosaic | 10 |

数据增强包括 HSV 颜色扰动、随机平移、尺度变化、水平翻转和 Mosaic。织物图像中颜色、光照和缺陷尺度变化明显，因此适当的数据增强可以提高模型泛化能力。

训练命令示例：

```bash
yolo detect train \
  model=yolo26n.pt \
  data=yolo_8_2/dataset.yaml \
  epochs=100 \
  imgsz=640 \
  batch=88 \
  optimizer=SGD \
  name=cloth_defect_run
```

### 4.3 验证与预测

训练完成后，使用验证集评估模型，并输出 Precision、Recall、mAP50、mAP50-95、PR 曲线、混淆矩阵和预测可视化图片。

验证命令示例：

```bash
yolo detect val \
  model=runs-MLworks/runs/detect/cloth_defect_run-5/weights/best.pt \
  data=yolo_8_2/dataset.yaml \
  imgsz=640 \
  conf=0.25 \
  iou=0.7
```

预测命令示例：

```bash
yolo detect predict \
  model=runs-MLworks/runs/detect/cloth_defect_run-5/weights/best.pt \
  source=yolo_8_2/images/val \
  imgsz=640 \
  conf=0.25
```

## 五、编程实现代码

下面代码展示了本实验的主要实现流程，包括数据转换、训练、验证和预测。实际转换脚本见 `convert_to_yolo.py`。

```python
from pathlib import Path
import json
import random
import shutil

from ultralytics import YOLO


BASE_DIR = Path("D:/Study/rh/workspace/machineVisionWork-0511/data/"
                "smartdiagnosisofclothflaw_round1train1_datasets_partA/partA")
DEFECT_DIR = BASE_DIR / "org-data/defect_Images"
NORMAL_DIR = BASE_DIR / "org-data/normal_Images"
ANNO_PATH = BASE_DIR / "org-data/Annotations/anno_train.json"
OUT_DIR = BASE_DIR / "yolo_8_2"

NAME_TO_ID = {
    "破洞": 1, "水渍": 2, "油渍": 2, "污渍": 2, "三丝": 3, "结头": 4,
    "花板跳": 5, "百脚": 6, "毛粒": 7, "粗经": 8, "松经": 9, "断经": 10,
    "吊经": 11, "粗维": 12, "纬缩": 13, "浆斑": 14, "整经结": 15,
    "星跳": 16, "跳花": 16, "断氨纶": 17, "稀密档": 18, "浪纹档": 18,
    "色差档": 18, "磨痕": 19, "轧痕": 19, "修痕": 19, "烧毛痕": 19,
    "死皱": 20, "云织": 20, "双维": 20, "双纬": 20, "双经": 20,
    "跳纱": 20, "筘路": 20, "纬纱不良": 20,
}


def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return xc, yc, bw, bh


def convert_dataset(image_size=(2446, 1000), train_ratio=0.8, seed=42):
    annotations = json.loads(ANNO_PATH.read_text(encoding="utf-8"))
    image_to_annos = {}
    for item in annotations:
        image_to_annos.setdefault(item["name"], []).append(item)

    images = []
    for p in DEFECT_DIR.glob("*.jpg"):
        images.append((p.name, p))
    for p in NORMAL_DIR.glob("*.jpg"):
        images.append((p.name, p))

    random.Random(seed).shuffle(images)
    split = int(len(images) * train_ratio)
    groups = {"train": images[:split], "val": images[split:]}

    for split_name, split_images in groups.items():
        (OUT_DIR / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for image_name, image_path in split_images:
            shutil.copy2(image_path, OUT_DIR / "images" / split_name / image_name)
            label_path = OUT_DIR / "labels" / split_name / image_name.replace(".jpg", ".txt")

            lines = []
            for anno in image_to_annos.get(image_name, []):
                cls_id = NAME_TO_ID[anno["defect_name"]]
                xc, yc, bw, bh = xyxy_to_yolo(anno["bbox"], *image_size)
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # 正常图写空 txt，作为负样本。
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def train_and_validate():
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(OUT_DIR / "dataset.yaml"),
        epochs=100,
        imgsz=640,
        batch=88,
        optimizer="SGD",
        name="cloth_defect_run",
    )
    model.val(data=str(OUT_DIR / "dataset.yaml"), imgsz=640, iou=0.7)


def predict_one_image(image_path):
    model = YOLO("runs-MLworks/runs/detect/cloth_defect_run-5/weights/best.pt")
    results = model.predict(source=image_path, imgsz=640, conf=0.25)
    return results
```

## 六、实验结果

训练结果目录为：

```text
runs-MLworks/runs/detect/cloth_defect_run-5
```

最优结果出现在第 90 个 epoch：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.31639 |
| Recall | 0.30548 |
| mAP50 | 0.28824 |
| mAP50-95 | 0.13804 |
| val box loss | 2.16451 |
| val cls loss | 2.42032 |
| val dfl loss | 0.01519 |

第 100 个 epoch 的最终结果为：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.30627 |
| Recall | 0.30045 |
| mAP50 | 0.27619 |
| mAP50-95 | 0.12728 |
| val box loss | 2.21407 |
| val cls loss | 2.45241 |
| val dfl loss | 0.01475 |

### 6.1 标签分布

![标签分布](runs-MLworks/runs/detect/cloth_defect_run-5/labels.jpg)

从标签分布图可以看出，疵点框尺寸差异明显，且类别数量不均衡。部分类别样本较少，这会影响模型对少数类别的学习。

### 6.2 训练曲线

![训练曲线](runs-MLworks/runs/detect/cloth_defect_run-5/results.png)

训练过程中 box loss、cls loss 整体下降，说明模型逐渐学习到了织物缺陷的位置和类别特征。但验证集指标提升有限，说明模型仍存在漏检和误检问题。

### 6.3 PR 曲线

![PR 曲线](runs-MLworks/runs/detect/cloth_defect_run-5/BoxPR_curve.png)

PR 曲线反映了不同置信度阈值下 Precision 与 Recall 的变化。曲线面积不高，说明在当前模型和数据规模下，模型对多类别疵点的稳定区分能力仍然不足。

### 6.4 F1 曲线

![F1 曲线](runs-MLworks/runs/detect/cloth_defect_run-5/BoxF1_curve.png)

F1 曲线用于综合观察查准率和查全率的平衡。实际使用时可以根据生产需求调整置信度阈值：如果希望尽量减少漏检，可以适当降低阈值；如果希望减少误报，则应提高阈值。

### 6.5 混淆矩阵

![混淆矩阵](runs-MLworks/runs/detect/cloth_defect_run-5/confusion_matrix.png)

![归一化混淆矩阵](runs-MLworks/runs/detect/cloth_defect_run-5/confusion_matrix_normalized.png)

混淆矩阵显示，不同疵点类别之间存在混淆，尤其是纹理相近、外观相似的类别更容易被误分。例如水渍、油渍、污渍在 README 中被归并到同一个 category id，本身就具有视觉相似性；磨痕、轧痕、修痕、烧毛痕也被归并到同一类，模型更关注共同视觉特征。

### 6.6 验证集预测效果

验证集标注与预测对比如下：

![验证集真实标签 0](runs-MLworks/runs/detect/cloth_defect_run-5/val_batch0_labels.jpg)

![验证集预测结果 0](runs-MLworks/runs/detect/cloth_defect_run-5/val_batch0_pred.jpg)

![验证集真实标签 1](runs-MLworks/runs/detect/cloth_defect_run-5/val_batch1_labels.jpg)

![验证集预测结果 1](runs-MLworks/runs/detect/cloth_defect_run-5/val_batch1_pred.jpg)

从预测图可以看到，模型能够检测出部分明显疵点，并给出对应类别和检测框。但对于细长、小面积、与背景纹理对比度低的缺陷，检测框容易偏移或漏检。

## 七、性能分析与误差原因

本实验模型在验证集上最优 mAP50 为 0.28824，mAP50-95 为 0.13804，说明模型已经具备一定的疵点定位能力，但整体性能仍有较大提升空间。主要原因如下：

1. 疵点目标小且形态不规则。织物缺陷常表现为细线、斑点或局部纹理异常，在 640 输入尺寸下容易被缩小，导致细节信息不足。
2. 类别不均衡明显。类别 4 有 782 个框，类别 5、6 只有 50 多个框，少数类别难以充分学习。
3. 类间视觉相似度高。水渍、油渍、污渍等类别外观接近；部分结构性缺陷也和织物纹理方向相似，容易误判。
4. 背景纹理干扰强。织物本身存在周期性纹理，模型可能把正常纹理变化误识别为缺陷，也可能把真实缺陷看作背景纹理的一部分。
5. 数据集存在不完整情况。当前文件夹中有部分 JSON 标注对应的图片缺失，实际训练数据少于完整标注规模，影响模型泛化能力。
6. 模型规模较小。`yolo26n.pt` 属于轻量模型，推理速度快，但特征表达能力有限，对复杂细粒度缺陷不如更大的模型。

后续改进方向：

1. 使用更高输入分辨率，例如 `imgsz=1024` 或切片检测，以保留小缺陷细节。
2. 使用更大模型或专门针对小目标优化的检测结构。
3. 对少数类别进行重采样、类别均衡增强或增加数据。
4. 引入织物纹理增强、局部对比度增强、缺陷区域裁剪等预处理。
5. 根据实际生产需求调节置信度阈值，在漏检率和误检率之间取得合适平衡。
6. 补全缺失图片后重新训练，使用完整数据提高泛化能力。

## 八、课程总结、学习体会与感想

通过本次课程实验，我对机器视觉系统的完整流程有了更清晰的认识。一个视觉检测任务并不只是训练模型，还包括理解数据、清洗标注、设计类别映射、划分训练验证集、选择合适算法、观察训练曲线、分析误差并提出改进方案。

在数据处理阶段，我体会到标注格式和数据一致性非常重要。即使模型算法本身很先进，如果图片和标签无法正确对应，训练结果也会受到明显影响。将 JSON 标注转换为 YOLO 格式的过程，让我理解了目标检测中边界框坐标、归一化和类别编号的实际含义。

在模型训练阶段，我认识到深度学习方法相比传统图像处理更适合复杂纹理和多类别缺陷检测，但它也依赖足够的数据量、合理的参数设置和有效的数据增强。实验结果虽然能够检测出一部分明显疵点，但指标并不算高，这说明工业视觉任务往往比普通图像分类更困难，特别是小目标、弱纹理差异和类别不均衡会显著影响检测效果。

总体来说，本课程让我从理论走向实践，理解了机器视觉在工业质检中的应用价值，也认识到算法落地需要兼顾准确率、速度、数据质量和误差分析。后续如果继续改进，我会优先尝试提高输入分辨率、补全数据、增强小目标样本，并比较不同 YOLO 模型规模对检测效果的影响。
