from ultralytics import YOLO

# ==== 配置路径 ====
DATA_YAML = 'data/BCCD-yolo/bccd.yaml'  # 训练数据配置
MODEL_NAME = 'yolo11n.pt'  # 可选: 'yolo11n.pt' 或 'yolo11s.pt' 等
RESULT_DIR = 'runs/detect/BCCD'  # 结果保存目录
EXP_NAME = 'BCCD_yolo11n'  # 实验名称
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16

# ==== Step 1: 加载模型 ====
model = YOLO(MODEL_NAME)  # 加载预训练模型

# ==== Step 2: 训练 ====
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    name='BCCD_yolo11n',  # 实验名称
    project=RESULT_DIR,
    val = False,  # 训练时不进行验证
    verbose=True
)

# ==== Step 3: 验证（验证集） ====
metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, verbose=True, exist_ok=True)
print("✅ 验证完成. mAP50:", metrics.box.map50)

# ==== Step 4: 测试（推理 test set） ====
model_path = f'{RESULT_DIR}/BCCD_yolo11n/weights/best.pt'  # 最佳模型路径
model = YOLO(model_path)

# 推理 test set 并保存结果图像
model.predict(
    source='data/BCCD-yolo/images/test',  # 测试图像路径
    imgsz=IMG_SIZE,
    conf=0.25,  # 置信度阈值
    save=True,  # 保存预测图像（带框）
    save_txt=True,  # 保存预测框为 txt
    project=RESULT_DIR,
    name='BCCD_test_infer',
    exist_ok=True
)

print("✅ 推理完成，结果保存在:", f"{RESULT_DIR}/BCCD_test_infer/")
