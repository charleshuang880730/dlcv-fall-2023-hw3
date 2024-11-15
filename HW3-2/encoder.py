import timm
import torch
from torch import nn
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"

class ViT(nn.Module):

    def __init__(self, vit_model_name='vit_large_patch16_224', out_features=768):
        super().__init__()
        # 加載預訓練的 ViT 模型
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        self.vit.head = nn.Identity()
        self.additional_linear = nn.Linear(1280, out_features)

    def forward(self, x):
        # 從 ViT 獲得視覺特徵
        with torch.no_grad():
            visual_features = self.vit(x)  # [batch_size, num_patches, vit_features_dim]
        # visual_features = self.additional_linear(visual_features)

        return visual_features

class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.model, _ = clip.load('ViT-L/14', device=device)
        
    def forward(self, x):
        with torch.no_grad():
            visual_features = self.model.encode_image(x)
        return visual_features

"""available_vit_huge_models = [model for model in timm.list_models() if 'vit_large' in model]
print(available_vit_huge_models)
print(clip.available_models())"""