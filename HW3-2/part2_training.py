import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor
import torch
import torch.nn as nn
from encoder import ViT, CLIP
from decoder import Decoder, Config
from tokenizer import BPETokenizer
from tqdm.auto import tqdm


myseed = 666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

tokenizer = BPETokenizer('./HW3-2/encoder.json', './HW3-2/vocab.bpe')

def prepare_gt_tokens(targets, start_token_id=50256, end_token_id=50256, pad_token_id=50256, ignore_index=-100):

    targets_modify = targets[:, 1:].clone()

    for i in range(targets_modify.size(0)):
        end_token_pos = (targets_modify[i] == end_token_id).nonzero(as_tuple=True)[0]
        if end_token_pos.nelement() != 0:

            end_token_pos = end_token_pos[0].item()
            if end_token_pos + 1 < targets_modify.size(1):
                targets_modify[i, end_token_pos + 1:] = ignore_index

    targets_modify = torch.nn.functional.pad(targets_modify, (0, 1), value=ignore_index)

    return targets_modify
    

class ImageCaptionDataset(Dataset):
    def __init__(self, data_folder, annotation_file, transform=None, tokenizer=None, max_len=59):
        """
        Args:
            data_folder (string): 路徑到包含圖像的文件夾。
            annotation_file (string): 包含標題的 JSON 文件的路徑。
            transform (callable, optional): 一組轉換應用於圖像。
            tokenizer (BPETokenizer, optional): 分詞器用於處理標題。
            max_len (int, optional): 分詞後標題的最大長度。
        """
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.data_folder = data_folder
        self.transform = transform if transform else Compose([Resize((224, 224)), ToTensor()])
        self.tokenizer = tokenizer if tokenizer else BPETokenizer('./encoder.json', './vocab.bpe')
        self.max_len = max_len
        self.samples = []

        for img in self.annotations['images']:
            img_id = img['id']
            img_file = os.path.join(data_folder, img['file_name'])
            img_captions = [ann['caption'] for ann in self.annotations['annotations'] if ann['image_id'] == img_id]
            
            for caption in img_captions:
                self.samples.append((img_file, caption))
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
    
        img_file, caption = self.samples[idx]
        image = Image.open(img_file).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        tokenize_word = self.tokenizer.encode(caption)

        word = [50256] + tokenize_word + [50256] * (self.max_len - len(tokenize_word))
        word = word[:self.max_len]

        return image, torch.tensor(word, dtype=torch.long)

# 數據轉換，例如 resize、to tensor 和 normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 根據你的模型和訓練數據選擇適當的 mean 和 std。
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

batch_size = 64
n_epochs = 5
patience = 15
save_interval = 1  # 每 1 個 epoch 儲存一次模型

device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config(checkpoint="../hw3_data/p2_data/decoder_model.bin")
# encoder = ViT().to(device)
encoder = CLIP().to(device)
# checkpoint = torch.load("p2_adapter_timm_epoch_5.bin")
decoder = Decoder(config).to(device)
# decoder.load_state_dict(checkpoint)
decoder.freeze_pretrained_layers()
print("Total params: ", sum(p.numel () for p in decoder.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 創建數據集和數據加載器
train_dataset = ImageCaptionDataset(
    data_folder='../hw3_data/p2_data/images/train',
    annotation_file='../hw3_data/p2_data/train.json',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

named_parameters = dict(decoder.named_parameters())

completed_epochs = 0

for epoch in range(completed_epochs, n_epochs):

    train_loss = []
    best_train_loss = 100
    max_seq_length = 59
    decoder.train()

    for images, captions in tqdm(train_loader):

        images = images.to(device)

        captions = captions.to(device)

        image_features = encoder(images).float().to(device)
        
        caption_output = decoder(captions, image_features)

        caption_output = caption_output.view(-1, caption_output.size(-1))  # [batch_size * sequence_length, num_classes]

        adjusted_captions = prepare_gt_tokens(captions) # GT

        adjusted_captions = adjusted_captions.view(-1)

        loss = criterion(caption_output, adjusted_captions)
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = sum(train_loss) / len(train_loss)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(decoder.state_dict(), 'p2_adapter_clip_large3_best.bin')
        print('Save model with loss:', best_train_loss)

        # 過濾出未被凍結的參數
        # trainable_params = {k: v for k, v in decoder.state_dict().items() if named_parameters()[k].requires_grad}
        # 儲存這些參數
        # torch.save(trainable_params, 'p2_adapter_timm_train_only.bin')

    if (epoch + 1) % save_interval == 0:
        # 儲存模型
        torch.save(decoder.state_dict(), f'p2_adapter_clip_large3_epoch_{epoch + 1}.bin')
        print(f'Model saved at epoch {epoch + 1}')

        # trainable_params = {k: v for k, v in decoder.state_dict().items() if named_parameters()[k].requires_grad}
        # 儲存這些參數
        # torch.save(trainable_params, f'p2_adapter_epoch_{epoch + 1}_train_only.bin')

    scheduler.step()
    