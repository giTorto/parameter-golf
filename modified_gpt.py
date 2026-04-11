import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from rmsnorm import RMSNorm
from causal_self_attention import CausalSelfAttention
from mlp import MLP
import torch.nn.functional as F
from find_ideal_byte_length import build_byte_table



class DictionaryFactory(nn.Module):

    def __init__(self, vocab_size: int, max_bytes: int=8, bytes_dim: int=16, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 1. The Barcode Maker (260 slots)
        # 0 = Padding
        # 1-256 = Standard text + Byte Fallbacks
        # 257-259 = Control Tokens (<s>, </s>, <unk>)
        self.byte_embedding = nn.Embedding(260, bytes_dim, padding_idx=0)

        # 2. simplified H-Net, stretching bytes to d_model
        self.stretching_engine = nn.Linear(max_bytes * byte_dim, d_model)

        # 1. Initialize the byte embeddings with normal distribution
        nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.005)
        
        #2. Force the Stretching Engine to generate a "quiet" dictionary
        tied_std = d_model ** -0.5
        nn.init.normal_(self.stretching_engine.weight, mean=0.0, std=tied_std)

        # 3. Zero out the bias so it doesn't skew the dictionary off-center
        nn.init.zeros_(self.stretching_engine.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        # 1. Get the byte embeddings
        byte_embeddings = self.byte_embedding(input_ids)

        # Flatten the 8 bytes into a single 128-length array per word
        flattened = barcodes.view(self.vocab_size, -1)        
        
        # 2. Stretch the bytes to d_model
        stretched_bytes = self.stretching_engine(flattened)
        return stretched_bytes

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
        self.factory = DictionaryFactory(vocab_size, model_dim)
        self.cheat_sheet = cheat_sheet
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.byte_table = build_byte_table(vocab_size, model_dim)
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
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
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
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")