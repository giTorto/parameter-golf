import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from find_ideal_byte_length import BYTE_BARCODE_VOCAB_SIZE, analyze_vocab_lengths, build_byte_table
from transformer_layers import CastedLinear, CausalSelfAttention, MLP, RMSNorm


class GeluDictionaryFactory(DictionaryFactory):
    def __init__(self, vocab_size: int, max_bytes: int=8, bytes_dim: int=16, d_model: int=384):
        self.vocab_size = vocab_size
        self.tied_std = d_model ** -0.5

        # 1. The Barcode Maker (260 slots)
        # 0 = Padding
        # 1-256 = Standard text + Byte Fallbacks
        # 257-259 = Control Tokens (<s>, </s>, <unk>, <pad>)
        self.byte_embedding = nn.Embedding(261, bytes_dim, padding_idx=0)

        self.stretching_engine_f1 = nn.Linear(max_bytes * bytes_dim, hidden_dim) 
        self.gelu = nn.GeLU()
        self.stretching_engine_f2 = nn.Linear(hidden_dim, d_model)

        # 1. Initialize the byte embeddings with normal distribution
        nn.init.kaiming_normal_(self.stretching_engine_f1.weight, mode='fan_in', nonlinearity='leaky_relu')
        
        #2. Force the Stretching Engine to generate a "quiet" dictionary
        nn.init.normal_(self.stretching_engine_f2.weight, mean=0.0, std=self.tied_std)

        # 3. Zero out the bias so it doesn't skew the dictionary off-center
        nn.init.zeros_(self.stretching_engine_f1.bias)
        nn.init.zeros_(self.stretching_engine_f2.bias)

    def forward(self, byte_indices: Tensor) -> Tensor:
        # byte_indices: (vocab_size, max_bytes) token indices on same device as module
        byte_embeddings = self.byte_embedding(byte_indices)
        flattened = byte_embeddings.view(byte_indices.size(0), -1)
        stretched_bytes = self.stretching_engine_f1(flattened)
        stretched_bytes = self.gelu(stretched_bytes)
        stretched_bytes = self.stretching_engine_f2(stretched_bytes)
        return stretched_bytes



class DictionaryFactory(nn.Module):

    def __init__(self, vocab_size: int, max_bytes: int=8, bytes_dim: int=16, d_model: int=384):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 1. The Barcode Maker (260 slots)
        # 0 = Padding
        # 1-256 = Standard text + Byte Fallbacks
        # 257-259 = Control Tokens (<s>, </s>, <unk>, <pad>)
        self.byte_embedding = nn.Embedding(261, bytes_dim, padding_idx=0)

        # 2. simplified H-Net, stretching bytes to d_model
        # original stretching engine 
        self.stretching_engine = nn.Linear(max_bytes * bytes_dim, d_model)

        # 1. Initialize the byte embeddings with normal distribution
        nn.init.normal_(self.byte_embedding.weight, mean=0.0, std=0.005)
        
        #2. Force the Stretching Engine to generate a "quiet" dictionary
        self.tied_std = d_model ** -0.5
        nn.init.normal_(self.stretching_engine.weight, mean=0.0, std=self.tied_std)

        # 3. Zero out the bias so it doesn't skew the dictionary off-center
        nn.init.zeros_(self.stretching_engine.bias)

    def forward(self, byte_indices: Tensor) -> Tensor:
        # byte_indices: (vocab_size, max_bytes) token indices on same device as module
        byte_embeddings = self.byte_embedding(byte_indices)
        flattened = byte_embeddings.view(byte_indices.size(0), -1)
        return self.stretching_engine(flattened)

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class ModifiedGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        max_bytes = analyze_vocab_lengths("./data/tokenizers/fineweb_1024_bpe.model")
        bytes_dim = 16
        self.factory = DictionaryFactory(vocab_size, max_bytes, bytes_dim=bytes_dim, d_model=model_dim)
        # Buffer so .to(device) / DDP keep indices on the same device as embeddings (required for compile).
        self.register_buffer("byte_table", build_byte_table("./data/tokenizers/fineweb_1024_bpe.model", max_bytes))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.factory.byte_embedding.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:

        ghost_dict = self.factory(self.byte_table)

        x = F.embedding(input_ids, ghost_dict)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, ghost_dict)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")