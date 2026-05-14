from ultralytics import YOLO

def train_cloth_model():
    # 1. 加载预训练模型 (建议使用 YOLOv8n 或 v11n 以兼顾速度与工业部署)
    # 如果追求更高精度，可以将 'n' 改为 's' 或 'm'
    model = YOLO('yolo26n.pt') 

    # 2. 开始训练
    results = model.train(
        data='/root/autodl-tmp/data/dataset.yaml',      # 指定数据集配置
        epochs=100,                  # 训练轮数
        imgsz=640,                   # 输入图像尺寸
        batch=88,                    # 批次大小，视显存大小调整
        device=0,                    # 使用 GPU 0 运行
        workers=8,                   # 数据加载线程数
        name='cloth_defect_run',     # 实验名称
        pretrained=True,             # 使用预训练权重
        optimizer='SGD',             # 优化器
        lr0=0.01,                    # 初始学习率
        augment=True,                # 开启在线数据增强
        rect=False                   # 如果布匹图像多为固定比例，可设为 True 提高效率
    )

    # 3. 验证模型
    metrics = model.val()
    print(f"训练完成，验证集 mAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    train_cloth_model()