import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, hp):
        super(Embedding, self).__init__()

        self.words_embed = nn.Embedding(hp.vocab_size, hp.d_model)
        self.positions_embed = nn.Embedding(hp.max_position_embeddings, hp.d_model)
        self.dropout = nn.Dropout(hp.dropout_rate)

        self.register_buffer("position_ids", torch.arange(hp.max_position_embeddings))

    def forward(self, input_ids):
        input_shape = input_ids.size()
        position_ids = self.position_ids[None, :input_shape[-1]]

        inputs_embed = self.words_embed(input_ids)
        position_embed = self.positions_embed(position_ids)

        x = inputs_embed + position_embed

        out = self.dropout(x)
        return out


class Attention(nn.Module):
    def __init__(self, hp):
        super(Attention, self).__init__()
        assert hp.d_model % hp.n_head == 0
        self.n_head = hp.n_head
        self.d_model = hp.d_model
        self.d_k = hp.d_model // hp.n_head

        self.W_QKV = nn.Linear(hp.d_model, hp.d_model*3)
        self.WO = nn.Linear(hp.d_model, hp.d_model)

        self.attn_dropout = nn.Dropout(hp.dropout_rate)
        self.proj_dropout = nn.Dropout(hp.dropout_rate)

    def split_heads(self, x):
        input_shape = x.size()
        x = x.view([-1, input_shape[1], self.n_head, self.d_k])
        return x.permute(0, 2, 1, 3)

    def attn(self, q, k, v, mask):
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.d_k)
        if mask is not None:
            qk = qk * (1 - mask) + mask * -1e9
        qk = F.softmax(qk, dim=-1)
        qk = self.attn_dropout(qk)

        attn_value = torch.matmul(qk, v)
        return attn_value

    def forward(self, x, mask=None):
        input_shape = x.size()
        x = self.W_QKV(x)
        query, key, value = x.split(self.d_model, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attn_value = self.attn(query, key, value, mask)
        attn_value = attn_value.permute(0, 2, 1, 3).contiguous().view(input_shape)

        out = self.WO(attn_value)
        out = self.proj_dropout(out)

        return out


class MLP(nn.Module):
    def __init__(self, hp):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hp.d_model, hp.d_ff)
        self.fc2 = nn.Linear(hp.d_ff, hp.d_model)
        self.dropout = nn.Dropout(hp.dropout_rate)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        out = self.dropout(x)
        return out


class Block(nn.Module):
    def __init__(self, hp):
        super(Block, self).__init__()
        self.d_model = hp.d_model
        self.attn = Attention(hp)
        self.ln1 = nn.LayerNorm(hp.d_model, eps=hp.layer_norm_epsilon)
        self.mlp = MLP(hp)
        self.ln2 = nn.LayerNorm(hp.d_model, eps=hp.layer_norm_epsilon)

    def forward(self, x, mask=None):
        a = self.attn(x, mask=mask)
        l = self.ln1(x+a)
        m = self.mlp(l)
        out = self.ln2(l+m)
        return out


class OpenAIGPTModel(nn.Module):
    def __init__(self, hp, pad_token_id):
        super(OpenAIGPTModel, self).__init__()

        self.embed = Embedding(hp)
        self.layers = nn.ModuleList([Block(hp) for _ in range(hp.n_layer)])
        self.pad_token_id = pad_token_id

    def forward(self, input_ids):
        seq_len = input_ids.size(-1)
        padding_mask = (input_ids == self.pad_token_id)
        look_ahead_mask = 1 - torch.tril(torch.ones(seq_len, seq_len).view(1, 1, seq_len, seq_len))

        x = self.embed(input_ids)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).to(x.dtype).to(x.device)
        look_ahead_mask = look_ahead_mask.to(x.dtype).to(x.device)

        mask = torch.max(padding_mask, look_ahead_mask)
        for i, block in enumerate(self.layers):
            x = block(x, mask=mask)
        return x

