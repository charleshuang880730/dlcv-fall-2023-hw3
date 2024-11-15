import os
import torch
import clip
from PIL import Image
import pandas as pd
import json
from tqdm import tqdm
import argparse

# 設定命令行參數
argparser = argparse.ArgumentParser()
argparser.add_argument("--input_folder", type=str, default="./hw3_data/p1_data/val/", help="Input folder path containing the images.")
argparser.add_argument("--output", type=str, default="p1_pred.csv", help="Output file path.")
argparser.add_argument("--id2label", type=str, default="./hw3_data/p1_data/id2label.json", help="Path to the id2label JSON file.")
args = argparser.parse_args()

# 加載預訓練的 CLIP 模型和處理流程
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# 加載 id 到 label 的映射
with open(args.id2label) as f:
    id2label = json.load(f)

# 根據 id2label 創建提示文本
prompts = [f"This is a photo of {label}." for label in id2label.values()]

# 將提示文本轉換為 CLIP 可處理的格式
text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)

# 設置 Dataset 類別來處理圖片和獲取標籤
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, preprocess, id2label):
        self.data_folder = data_folder
        self.image_paths = sorted([os.path.join(data_folder, fname) for fname in os.listdir(data_folder) if fname.endswith('.png')])
        self.preprocess = preprocess
        self.id2label = id2label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        processed_image = self.preprocess(image)
        # 從檔案名稱中獲取類別標籤
        # class_label = os.path.basename(image_path).split('_')[0]
        # class_name = self.id2label[class_label]
        return processed_image, image_path

# 創建數據集和數據加載器
dataset = CLIPDataset(args.input_folder, preprocess, id2label)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# 進行預測並保存結果
predictions = []
model.eval()
with torch.no_grad():
    for images, image_paths in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_inputs)

        # 計算圖像和文字特徵之間的相似度
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=-1)
        pred_classes = probs.argmax(dim=-1)

        # 從 CLIP 的預測中獲取預測的類別標籤
        pred_labels = [id2label[str(pred.item())] for pred in pred_classes]

        predictions.extend(zip([os.path.basename(path) for path in image_paths], pred_labels))

# 將預測結果保存到 CSV
df = pd.DataFrame(predictions, columns=['filename', 'label'])
df.to_csv(args.output, index=False)
print(f"Predictions saved to {args.output}")


# 計算準確度
"""correct_predictions = sum([pred == true for _, pred, true in predictions])
total_predictions = len(predictions)
accuracy = correct_predictions / total_predictions
print(f"Accuracy over all test images: {accuracy:.4f}")"""