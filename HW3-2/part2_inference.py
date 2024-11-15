import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor
import torch
import torch.nn as nn
import numpy as np
from encoder import ViT, CLIP
from decoder import Decoder, Config
from tokenizer import BPETokenizer
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import argparse

# 設定命令行參數
argparser = argparse.ArgumentParser()
argparser.add_argument("--input_folder", type=str, default="./hw3_data/p1_data/val/", help="Input folder path containing the images.")
argparser.add_argument("--output", type=str, default="p1_pred.csv", help="Output file path.")
argparser.add_argument("--model", type=str, default="./hw3_data/p1_data/id2label.json", help="Path to the id2label JSON file.")
args = argparser.parse_args()

myseed = 666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

tokenizer = BPETokenizer('./HW3-2/encoder.json', './HW3-2/vocab.bpe')

def beam_search(decoder, image_features, tokenizer, beam_size=3, max_len=59, start_token=50256, end_token=50256):
    # 初始化 beam
    init_seq = torch.full((beam_size, 1), start_token, dtype=torch.long, device=device)
    sequences = [[seq, 0] for seq in init_seq]  # 每個序列有其對應的分數

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue

            logits = decoder(seq.unsqueeze(0), image_features)  # 生成 logits
            logits_nt = logits[:, -1, :]  # 只需最後一個 token 的 logits
            probs = nn.functional.softmax(logits_nt, dim=-1)
            topk_probs, topk_indices = probs.topk(beam_size)

            for i in range(beam_size):
                nt = topk_indices[0][i]
                new_seq = torch.cat([seq, nt.unsqueeze(0)], dim=0)
                new_score = score + torch.log(topk_probs[0][i])
                all_candidates.append((new_seq, new_score))

        # 排序並選擇最佳候選序列
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    best_seq, _ = sequences[0]
    return best_seq.squeeze(0).cpu().numpy().tolist()

class ImageCaptionDataset(Dataset):
    def __init__(self, data_folder, annotation_file=None, transform=None, tokenizer=None, max_len=59):
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = None
        
        self.data_folder = data_folder
        self.transform = transform if transform else Compose([Resize((224, 224)), ToTensor()])
        self.tokenizer = tokenizer if tokenizer else BPETokenizer('./HW3-2/encoder.json', './HW3-2/vocab.bpe')
        self.max_len = max_len
        self.image_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        caption = ""
        if self.annotations:
            image_id = os.path.splitext(self.image_files[idx])[0]
            captions = [ann['caption'] for ann in self.annotations['annotations'] if str(ann['image_id']) == image_id]
            caption = captions[0] if captions else ""
        
        tokenized_word = self.tokenizer.encode(caption)
        word = [50256] + tokenized_word + [50256] * (self.max_len - len(tokenized_word))
        word = word[:self.max_len]

        filename_without_extension, _ = os.path.splitext(self.image_files[idx])

        return image, torch.tensor(word, dtype=torch.long), filename_without_extension

# 數據轉換，例如 resize、to tensor 和 normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config(checkpoint = args.model)
# encoder = ViT().to(device)
encoder = CLIP().to(device)
decoder = Decoder(config).to(device)
decoder.load_state_dict(torch.load("./p2_adapter_clip_large3_train.bin", map_location=device), strict=False)
decoder.freeze_pretrained_layers()
print("Total params: ", sum(p.numel () for p in decoder.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-5)

val_dataset = ImageCaptionDataset(
    data_folder=args.input_folder,
    # annotation_file='../hw3_data/p2_data/val.json',
    transform=transform
)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

problematic_tokens = {255, 3907, 8836, 16253, 18433, 20804, 22020, 25084, 27764, 29690, 29826, 34633, 36310, 39588, 40792, 41200, 48953, 49476}

with torch.no_grad():
    
    decoder.eval()  
    predictions = {} 
    start_token = 50256
    end_token = 50256
    max_len = 59

    fig, axs = plt.subplots(nrows=max_len, ncols=1, figsize=(5, 2 * max_len))

    for images, _, img_name in tqdm(val_loader):
        images = images.to(device)
        image_features = encoder(images).float().to(device)

        for i in range(image_features.size(0)):
            captions = torch.full((1, 1), start_token, dtype=torch.long, device=device)
            features = image_features[i].unsqueeze(0)
            
            for _ in range(max_len):

                logits = decoder(captions, features)
                logits_nt = logits[:, -1, :]
                nt = torch.argmax(logits_nt, dim=-1)

                if nt.item() in problematic_tokens:
                    # 如果最可能的 token 是有問題的，選擇第二適合的
                    sorted_probs, sorted_indices = torch.sort(logits_nt, descending=True)
                    for idx in sorted_indices[0]:
                        if idx.item() not in problematic_tokens:
                            nt = idx
                            break

                nt = nt.unsqueeze(-1)
                captions = torch.cat([captions, nt], dim=-1)

                if nt.item() == end_token:
                    break

            pred = captions.squeeze(0).cpu().numpy().tolist()

            tokenize_word = tokenizer.decode(pred).replace('<|endoftext|>', '').strip().split(tokenizer.decode([end_token]))[0]

            filename = img_name[i]

            predictions[str(filename)] = tokenize_word
        
    with open(args.output, 'w') as f:          
        json.dump(predictions, f)
        print("finish json.")
