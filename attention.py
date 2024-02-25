import torch
from torch import Tensor, BoolTensor
import torch.nn as nn

import math

class CasualAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        context_size: int,
        num_heads: int, 
        attention_dropuot_p: float = 0.1,
        out_dropout_p: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.num_heads = num_heads

        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.att_dropout = nn.Dropout(attention_dropuot_p)
        self.out_dropout = nn.Dropout(out_dropout_p)

        self.register_buffer('causal_mask',
            torch.triu(torch.ones([context_size, context_size],
                       dtype=torch.bool), diagonal=1)
                .view(1, 1, context_size, context_size))

    def forward(self, x: Tensor, mask: BoolTensor|None = None):
        B, C, H = x.shape

        x = self.Wqkv(x).reshape(B, C, 3, self.num_heads, H//self.num_heads)

        q, k, v = x.transpose(1, 3).unbind(dim=2)

        att = q @ k.transpose(-2, -1)
        att /= math.sqrt(H//self.num_heads)

        combined_mask = self.causal_mask[:, :, :C, :C]
        if mask is not None:
            combined_mask += mask.view(B, 1, 1, C)
        att.masked_fill_(combined_mask, float("-inf"))

        att = att.softmax(dim=-1)
        att = self.att_dropout(att)

        att = att @ v
        att = att.transpose(1, 2).reshape(B, C, H)

        out = self.out_dropout(self.Wo(att))

        return out
    
class FeedForward(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            expand_size: int,
            dropout_p: float = 0.1,
            activation: nn.Module = nn.GELU,
            bias: bool = True,
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)
        self.act = activation()
        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        
        x = self.dropout(self.fc2(x))

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            context_size: int,
            expand_size: int,
            num_heads: int, 
            ff_dropout_p: float = 0.1,
            attention_dropuot_p: float = 0.1,
            out_dropout_p: float = 0.1,
            activation: nn.Module = nn.GELU,
            attention: nn.Module = CasualAttention,
            ff_bias: bool = True,
            attention_bias: bool = True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)

        self.attention = attention(
            hidden_size = hidden_size,
            context_size = context_size,
            num_heads = num_heads, 
            attention_dropuot_p = attention_dropuot_p,
            out_dropout_p = out_dropout_p,
            bias = attention_bias,
        )

        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = FeedForward(
            hidden_size = hidden_size,
            expand_size = expand_size,
            dropout_p = ff_dropout_p,
            activation = activation,
            bias = ff_bias,
        )

    def forward(self, x: Tensor, mask: BoolTensor|None = None):
        x = x + self.attention(self.norm1(x), mask)

        x = x + self.ffn(self.norm2(x))

        return x
    
class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        context_size: int,
        expand_size: int,
        vocab_size: int,
        num_heads: int,
        num_layers: int,
        ff_dropout_p: float = 0.1,
        attention_dropuot_p: float = 0.1,
        out_dropout_p: float = 0.1,
        embed_dropout_p: float = 0.1,
        activation: nn.Module = nn.GELU,
        attention: nn.Module = CasualAttention,
        ff_bias: bool = True,
        attention_bias: bool = True,
        head_bias: bool = True,
        tie_weights: bool = False,
    ):
        super().__init__()
        
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(context_size, hidden_size)
        self.embed_dopout = nn.Dropout(embed_dropout_p)

        self.att_layers = nn.ModuleList([
            TransformerBlock(
                hidden_size = hidden_size,
                context_size = context_size,
                expand_size = expand_size,
                num_heads = num_heads,
                ff_dropout_p = ff_dropout_p,
                attention_dropuot_p = attention_dropuot_p,
                out_dropout_p = out_dropout_p,
                activation = activation,
                attention = attention,
                ff_bias = ff_bias,
                attention_bias = attention_bias,
            )
            for _ in range(num_layers)
        ])

        self.head_norm = nn.LayerNorm(hidden_size)

        self.head = nn.Linear(hidden_size, vocab_size, bias=head_bias)
                                
        if tie_weights:
            self.head.weight = self.vocab_embed.weight

        pos = torch.arange(0, context_size, dtype=torch.long)
        self.register_buffer('pos', pos, persistent=False)

        self.apply(self._init_weights)


    def forward(self, x: Tensor, mask: BoolTensor|None = None):
        x = self.vocab_embed(x) + self.pos_embed(self.pos[:x.shape[1]])
        x = self.embed_dopout(x)

        for block in self.att_layers:
            x = block(x, mask)

        x = self.head_norm(x)
        x = self.head(x)

        return x
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == 'fc2':
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)