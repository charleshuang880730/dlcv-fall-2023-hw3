import torch
from decoder import Decoder, Config

config = Config()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加載整個模型
decoder = Decoder(config).to(device)
decoder.load_state_dict(torch.load('p2_adapter_clip_large3_epoch_4.bin'), strict=False)

trainable_params = {k: v for k, v in decoder.named_parameters() if v.requires_grad}

# 儲存這些參數
torch.save(trainable_params, 'p2_adapter_clip_large3_train.bin')