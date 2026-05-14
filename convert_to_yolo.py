"""
Convert the Tianchi fabric defect PartA training set to YOLO txt format.

Output layout:
  yolo_8_2/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    dataset.yaml
"""
from __future__ import annotations

import json
import os
import random
import shutil
import struct
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFECT_DIR = BASE_DIR / "defect_Images"
NORMAL_DIR = BASE_DIR / "normal_Images"
ANNO_PATH = BASE_DIR / "Annotations" / "anno_train.json"
OUTPUT_DIR = BASE_DIR / "yolo_8_2"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

DEFECT_NAME_TO_CATEGORY = {
    "破洞": 1,
    "水渍": 2,
    "油渍": 2,
    "污渍": 2,
    "三丝": 3,
    "结头": 4,
    "花板跳": 5,
    "百脚": 6,
    "毛粒": 7,
    "粗经": 8,
    "松经": 9,
    "断经": 10,
    "吊经": 11,
    "粗维": 12,
    "纬缩": 13,
    "浆斑": 14,
    "整经结": 15,
    "星跳": 16,
    "跳花": 16,
    "断氨纶": 17,
    "稀密档": 18,
    "浪纹档": 18,
    "色差档": 18,
    "磨痕": 19,
    "轧痕": 19,
    "修痕": 19,
    "烧毛痕": 19,
    "死皱": 20,
    "云织": 20,
    "双维": 20,
    "双纬": 20,
    "双经": 20,
    "跳纱": 20,
    "筘路": 20,
    "纬纱不良": 20,
}

CATEGORY_NAMES = {
    0: "normal",
    1: "hole",
    2: "stain",
    3: "three_threads",
    4: "knot",
    5: "skip_pattern",
    6: "centipede",
    7: "fuzz_ball",
    8: "thick_warp",
    9: "loose_warp",
    10: "broken_warp",
    11: "hanging_warp",
    12: "thick_weft",
    13: "weft_shrinkage",
    14: "size_spot",
    15: "warping_knot",
    16: "skip",
    17: "broken_spandex",
    18: "density_or_color_block",
    19: "mark",
    20: "misc_weave_defect",
}


def image_size(path: Path) -> tuple[int, int]:
    """Return JPEG/PNG dimensions without third-party dependencies."""
    with path.open("rb") as f:
        header = f.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", header[16:24])
        if header[:2] != b"\xff\xd8":
            raise ValueError(f"unsupported image format: {path}")

        f.seek(2)
        while True:
            marker = f.read(2)
            while marker and marker[0] != 0xFF:
                marker = marker[1:] + f.read(1)
            while marker and marker[1] == 0xFF:
                marker = marker[:1] + f.read(1)
            if len(marker) != 2:
                break

            marker_code = marker[1]
            if marker_code in (0xD8, 0xD9):
                continue
            size_bytes = f.read(2)
            if len(size_bytes) != 2:
                break
            segment_size = struct.unpack(">H", size_bytes)[0]
            if 0xC0 <= marker_code <= 0xCF and marker_code not in (0xC4, 0xC8, 0xCC):
                segment = f.read(5)
                height, width = struct.unpack(">HH", segment[1:5])
                return width, height
            f.seek(segment_size - 2, os.SEEK_CUR)

    raise ValueError(f"could not read image size: {path}")


def xyxy_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), img_w))
    y1 = max(0.0, min(float(y1), img_h))
    x2 = max(0.0, min(float(x2), img_w))
    y2 = max(0.0, min(float(y2), img_h))

    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    x_center = x1 + box_w / 2.0
    y_center = y1 + box_h / 2.0
    return x_center / img_w, y_center / img_h, box_w / img_w, box_h / img_h


def collect_images() -> dict[str, Path]:
    images: dict[str, Path] = {}
    for image_dir in (DEFECT_DIR, NORMAL_DIR):
        for path in image_dir.iterdir():
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if path.name in images:
                    raise ValueError(f"duplicate image name found: {path.name}")
                images[path.name] = path
    return images


def make_dirs() -> None:
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_label(label_path: Path, image_path: Path, annos: list[dict]) -> int:
    img_w, img_h = image_size(image_path)
    lines = []
    skipped = 0

    for anno in annos:
        defect_name = anno["defect_name"]
        if defect_name not in DEFECT_NAME_TO_CATEGORY:
            raise ValueError(f"unknown defect_name: {defect_name}")
        cls_id = DEFECT_NAME_TO_CATEGORY[defect_name]
        x_center, y_center, box_w, box_h = xyxy_to_yolo(anno["bbox"], img_w, img_h)
        if box_w <= 0 or box_h <= 0:
            skipped += 1
            continue
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return skipped


def write_dataset_yaml() -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in CATEGORY_NAMES.items())
    yaml_text = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

nc: 21
names:
{names}
"""
    (OUTPUT_DIR / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    print("Loading annotations...")
    annotations = json.loads(ANNO_PATH.read_text(encoding="utf-8"))

    image_to_annos: dict[str, list[dict]] = defaultdict(list)
    for anno in annotations:
        image_to_annos[anno["name"]].append(anno)

    images = collect_images()
    missing = sorted(set(image_to_annos) - set(images))
    if missing:
        print(f"Warning: skipping {len(missing)} annotated image names not found in this folder.")
        print(f"Example missing images: {missing[:5]}")

    all_items = sorted(images.items())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_items)

    train_count = int(len(all_items) * TRAIN_RATIO)
    splits = {
        "train": all_items[:train_count],
        "val": all_items[train_count:],
    }

    make_dirs()
    skipped_boxes = 0
    for split, split_items in splits.items():
        print(f"Writing {split}: {len(split_items)} images")
        for image_name, image_path in split_items:
            dst_image = OUTPUT_DIR / "images" / split / image_name
            dst_label = OUTPUT_DIR / "labels" / split / f"{Path(image_name).stem}.txt"
            shutil.copy2(image_path, dst_image)
            skipped_boxes += write_label(dst_label, image_path, image_to_annos.get(image_name, []))

    write_dataset_yaml()

    label_count = sum(1 for _ in (OUTPUT_DIR / "labels").rglob("*.txt"))
    defect_image_count = sum(1 for name in images if name in image_to_annos)
    normal_image_count = len(images) - defect_image_count
    print("Done.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images: {len(images)} total = {defect_image_count} defect + {normal_image_count} normal")
    print(f"Split: train {len(splits['train'])}, val {len(splits['val'])}")
    print(f"YOLO label files: {label_count}")
    print(f"Skipped invalid boxes: {skipped_boxes}")


if __name__ == "__main__":
    main()
