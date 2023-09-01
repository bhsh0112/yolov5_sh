import torch
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms
from models.yolo import Detect

# 加载YOLOv5模型
# model = torch.load('weed_corn.pt')
model=Detect(nc=2)
model_weights_path='weed_corn.pt'
model.load_state_path=(torch.load(model_weights_path))
model.eval()

# 打开摄像头
cap = cv2.VideoCapture(4) # 摄像头号需改为输入

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 将OpenCV图像格式转换为PIL图像格式
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 进行目标检测
    with torch.no_grad():
        results = model(image)

    # 指定目标类别和置信度阈值
    target_class = 'weed'
    target_class_index = model.names.index(target_class)
    confidence_threshold = 0.5

    target_indices = (results.pred[0]['class'] == target_class_index) & (results.pred[0]['score'] > confidence_threshold)
    target_boxes = results.pred[0]['boxes'][target_indices]
    target_scores = results.pred[0]['scores'][target_indices]

    # 找到置信度最高的目标
    if len(target_scores) > 0:
        highest_score_index = torch.argmax(target_scores)
        target_box = target_boxes[highest_score_index]

    # 在原始帧上绘制目标框
    x0, y0, x1, y1 = target_box.tolist()
    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

    # 显示帧
    cv2.imshow('Target Extraction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()