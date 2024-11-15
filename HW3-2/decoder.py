import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora


class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.key = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x, visual_features):
        visual_features = visual_features.unsqueeze(1).expand(-1, x.size(1), -1)
        q = self.query(x).view(x.size(0), x.size(1), self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = self.key(visual_features).view(visual_features.size(0), visual_features.size(1), self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = self.value(visual_features).view(visual_features.size(0), visual_features.size(1), self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)

        self.attention_map = att

        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.n_embd))

class Adapter(nn.Module):
    def __init__(self, input_size, down_size):
        super(Adapter, self).__init__()
        self.down = nn.Linear(input_size, down_size)
        self.up = nn.Linear(down_size, input_size)
        self.activation = nn.GELU()

    def forward(self, x):
        down_projected = self.down(x)
        activated = self.activation(down_projected)
        up_projected = self.up(activated)
        return x + up_projected 


class Block(nn.Module):

    def __init__(self, cfg, add_adapter=False, adapter_down_size=256):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.crossattn = CrossAttention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))

        self.add_adapter = add_adapter
        if self.add_adapter:
            self.adapter = Adapter(cfg.n_embd, adapter_down_size)

    def forward(self, x, visual_features=None):
        x = x + self.attn(self.ln_1(x))
        if visual_features is not None:
            x = x + self.crossattn(x, visual_features)
        x = x + self.mlp(self.ln_2(x))
        if self.add_adapter:
            x = self.adapter(x)
        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            # h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            h = nn.Sequential(*[Block(cfg, add_adapter=i >= cfg.n_layer - 2) for i in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

        # self.additional_layer = nn.Linear(1024, cfg.n_embd)

    def forward(self, x: Tensor, visual_features: Tensor, return_attention=False):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        # visual_features = self.additional_layer(visual_features)

        x = self.transformer.wte(x) + self.transformer.wpe(pos) # word token embedding + word position embedding
        for block in self.transformer.h:
            x = block(x, visual_features)
        x = self.lm_head(self.transformer.ln_f(x))

        if return_attention:
            # 提取最後一層的注意力權重
            attention_weights = self.transformer.h[-1].crossattn.attention_map
            # print(attention_weights.shape)
            return x, attention_weights

        return x

    def freeze_pretrained_layers(self):
        for name, param in self.named_parameters():
            if 'crossattn' not in name and 'adapter' not in name and 'ln_' not in name and 'additional_layer' not in name:
                param.requires_grad = False

