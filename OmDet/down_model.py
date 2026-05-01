#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Om_AI_Lab/omdet-turbo-swin-tiny-hf', cache_dir="models")

# python /home/ilstone/remote-projects/CVDemos/OmDet/down_model.py