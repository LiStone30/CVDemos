import requests
from PIL import Image
from transformers import AutoProcessor, OmDetTurboForObjectDetection

# ================= 1. 使用本地路径加载模型 =================
# 注意：路径末尾不要加斜杠，直接指向包含 config.json 的文件夹
local_model_path = "./models/Om_AI_Lab/omdet-turbo-swin-tiny-hf"

# 加载处理器和模型（首次加载会读取本地文件，不会联网下载）
processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=True)
model = OmDetTurboForObjectDetection.from_pretrained(local_model_path, local_files_only=True)

image_path = "/home/ilstone/remote-projects/CVDemos/OmDet/before.png"
image = Image.open(image_path).convert("RGB")   # 确保 RGB 格式
classes = ["A diamond-shaped UI button"]  # 👈 在这里指定你要识别的 UI 元素，例如 ["play button"]

# 3. 模型推理
inputs = processor(image, text=classes, return_tensors="pt")
outputs = model(**inputs)

# 4. 处理与输出结果
results = processor.post_process_grounded_object_detection(
    outputs,
    classes=classes,
    target_sizes=[image.size[::-1]],
    score_threshold=0.3,   # 置信度阈值，只显示得分高于此值的检测结果
    nms_threshold=0.3,     # 非极大值抑制阈值
)[0]

# 打印检测结果
for score, class_name, box in zip(results["scores"], results["classes"], results["boxes"]):
    box = [round(i, 1) for i in box.tolist()]
    print(f"Detected {class_name} with confidence {round(score.item(), 2)} at location {box}")