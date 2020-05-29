import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def make_mask_from_lens(x, lens):
    if lens is None:
        return None
    mask = torch.zeros(x.shape[:2])
    for i, j in enumerate(lens):
        mask[i,:j] = 1
    return mask.unsqueeze(-1)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # layer = EncoderLayer()
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TransformerEncoder(nn.Module):
    """
    A standard Encoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, src_embed):
        super(TransformerEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed # Embedding function
    
    def forward(self, src, src_lens):
        src_mask = make_mask_from_lens(src, src_lens)
        return self.encoder(self.src_embed(src), src_mask), src_lens


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, first_only=False):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.first_only = first_only

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.first_only:
            w, others = sublayer(self.norm(x))
            return x + self.dropout(w), others
        else:
            return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)[0])
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is None:
            past = [None] * len(self.layers)
        
        new_past = []
        for layer, layer_past in zip(self.layers, past):
            x, new_layer_past = layer(x, memory, src_mask, tgt_mask, layer_past)
            new_past.append(new_layer_past)
        return self.norm(x), new_past

class TransformerDecoder(nn.Module):
    """
    A standard Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, decoder, tgt_embed):
        super(TransformerDecoder, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed  # Embedding function
        
    def forward(self, memory, src_lens, tgt, tgt_mask, past=None):
        src_mask = make_mask_from_lens(memory, src_lens)
        past_len = 0
        if past is not None:
            if past[0] is not None:
                if past[0][0] is not None:
                    past_len = past[0][0].shape[-2]
        return self.decoder(self.tgt_embed(tgt, past_len=past_len), memory, src_mask, tgt_mask, past)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout, first_only=True),
                                       SublayerConnection(size, dropout, first_only=False),
                                       SublayerConnection(size, dropout, first_only=False)])
 
    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        "Follow Figure 1 (right) for connections."
        m = memory
        x, past = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, past))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask.transpose(-2, -1))[0])
        return self.sublayer[2](x, self.feed_forward), past

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    first_col = torch.zeros_like(tgt)
    first_col[:, 0] = 1
    tgt_mask = ((tgt != pad) | first_col.bool()).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)
    p_attn = F.softmax(scores, dim = -1)#.masked_fill(mask == 0, 0.0)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None, past=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        if past is not None:
            assert mask is None, "Using past state while auto-regressive decoding without mask."
            past_key, past_value = past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
            temp_past = key, value
        else:
            temp_past = None
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask.to(query.device) if type(mask)!=type(None) else mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), temp_past

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionConvFeedForward(nn.Module):
    "Implements Conv+FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionConvFeedForward, self).__init__()
        self.c_1 = nn.Conv1d(d_model, d_ff, kernel_size=3, stride=1, padding=1)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.c_1(x.transpose(-2, -1)).transpose(-2, -1))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, pretrained_matrix=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        if pretrained_matrix is not None:
            self.lut.weight.data = torch.tensor(pretrained_matrix)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Scale(nn.Module):
    def __init__(self, d_model):
        super(Scale, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        return x * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x, past_len=0):
        x = x + self.pe[:, past_len:past_len+x.size(1)].type_as(x)
        return self.dropout(x)

# def Transformer(N=6, d_model=1024, d_ff=2048, h=8, dropout=0.1):
#     "Helper: Construct a model from hyperparameters."
#     c = copy.deepcopy
#     attn = MultiHeadedAttention(h, d_model)
#     ff = PositionwiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)
#     model = EncoderDecoder(
#         Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
#         Decoder(DecoderLayer(d_model, c(attn), c(attn), 
#                              c(ff), dropout), N),
#         ScalePositionEmbedding(Scale(d_model), c(position)),
#         ScalePositionEmbedding(Scale(d_model), c(position)), None)
    
#     # This was important from their code. 
#     # Initialize parameters with Glorot / fan_avg.
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model

class ScalePositionEmbedding(nn.Module):
    def __init__(self, scale, position):
        super(ScalePositionEmbedding, self).__init__()
        self.add_module('0', scale)
        self.add_module('1', position)

    def __getitem__(self, idx):
        return self._modules[str(idx)]

    def forward(self, x, past_len=0):
        x = self[0](x)
        x = self[1](x, past_len)
        return x

def SpeechTransformerEncoder(N=12, d_model=256, d_ff=2048, h=4, dropout=0.1):
    '''Helper: Construct a model from hyperparameters. 
    This encoder does not include a word embeddding layer.
    The input tensors need to be word embeddings/speech features.'''
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout=0.0)
    ff = PositionConvFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = TransformerEncoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        ScalePositionEmbedding(Scale(d_model), c(position)))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def SpeechTransformerDecoder(N=6, d_model=256, d_ff=2048, h=4, dropout=0.1):
    '''Helper: Construct a model from hyperparameters.
    This decoder does not include a word embeddding layer.
    The input tensors need to be word embeddings/speech features.'''
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout=0.0)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = TransformerDecoder(
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        ScalePositionEmbedding(Scale(d_model), c(position)))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

