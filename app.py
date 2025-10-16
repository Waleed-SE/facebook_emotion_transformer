import streamlit as st
import torch
import torch.nn as nn
import math
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt_tab')

import urllib.request

MODEL_URL = "https://huggingface.co/waleed1224/facebook_emotion_transformer/resolve/main/best_model_512_2_2_0.1_4096.pt"
MODEL_PATH = "best_model_512_2_2_0.1_4096.pt"
import os
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.q_linear(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.out(out), attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out, cross_attn_weights = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x, cross_attn_weights

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        for layer in self.decoder_layers:
            tgt, _ = layer(tgt, src, src_mask, tgt_mask)
        return self.out(tgt)

@st.cache_resource
def load_model():
    checkpoint = torch.load('best_model_512_2_2_0.1_4096.pt', map_location='cpu')
    vocab = checkpoint['vocab']
    inv_vocab = checkpoint['inv_vocab']
    model = Transformer(len(vocab))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, vocab, inv_vocab

def encode(text, vocab):
    tokens = ['<bos>'] + word_tokenize(text.lower()) + ['<eos>']
    return [vocab.get(t, vocab['<unk>']) for t in tokens]

def decode(ids, inv_vocab):
    words = [inv_vocab.get(i, '<unk>') for i in ids]
    words = [w for w in words if w not in ['<pad>', '<bos>', '<eos>', '<unk>', '<sep>'] and not w.startswith('<emotion_')]
    return ' '.join(words)

def make_causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)

def generate(model, src_text, vocab, inv_vocab, method='greedy', beam_width=3, max_len=50):
    src_ids = torch.tensor([encode(src_text, vocab)])
    
    with torch.no_grad():
        enc = model.pos_enc(model.embedding(src_ids) * math.sqrt(model.d_model))
        for layer in model.encoder_layers:
            enc = layer(enc)
        
        if method == 'greedy':
            ys = torch.tensor([[vocab['<bos>']]])
            for _ in range(max_len):
                tgt_mask = make_causal_mask(ys.size(1))
                tgt_emb = model.pos_enc(model.embedding(ys) * math.sqrt(model.d_model))
                for layer in model.decoder_layers:
                    tgt_emb, _ = layer(tgt_emb, enc, tgt_mask=tgt_mask)
                logits = model.out(tgt_emb[:, -1, :])
                next_token = logits.argmax(dim=-1).unsqueeze(0)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == vocab['<eos>']:
                    break
            return decode(ys.squeeze(0).tolist(), inv_vocab)
        
        else:
            beams = [(torch.tensor([[vocab['<bos>']]]), 0.0)]
            for _ in range(max_len):
                new_beams = []
                for seq, score in beams:
                    if seq[0, -1].item() == vocab['<eos>']:
                        new_beams.append((seq, score))
                        continue
                    tgt_mask = make_causal_mask(seq.size(1))
                    tgt_emb = model.pos_enc(model.embedding(seq) * math.sqrt(model.d_model))
                    for layer in model.decoder_layers:
                        tgt_emb, _ = layer(tgt_emb, enc, tgt_mask=tgt_mask)
                    logits = model.out(tgt_emb[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)
                    top_probs, top_indices = log_probs.topk(beam_width)
                    for i in range(beam_width):
                        next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                        next_score = score + top_probs[0, i].item()
                        next_seq = torch.cat([seq, next_token], dim=1)
                        new_beams.append((next_seq, next_score))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                if all(seq[0, -1].item() == vocab['<eos>'] for seq, _ in beams):
                    break
            return decode(beams[0][0].squeeze(0).tolist(), inv_vocab)

st.title("🤖 Empathetic Chatbot")
st.markdown("Transformer-based empathetic conversational agent")

model, vocab, inv_vocab = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    emotion = st.selectbox("Emotion", ['afraid', 'angry', 'annoyed', 'anticipating', 'anxious', 'apprehensive', 'ashamed', 'caring', 'confident', 'content', 'devastated', 'disappointed', 'disgusted', 'embarrassed', 'excited', 'faithful', 'furious', 'grateful', 'guilty', 'hopeful', 'impressed', 'jealous', 'joyful', 'lonely', 'nostalgic', 'prepared', 'proud', 'sad', 'sentimental', 'surprised', 'terrified', 'trusting'])
    situation = st.text_area("Situation (optional)", "")
    method = st.radio("Decoding", ['greedy', 'beam'])
    beam_width = 3
    if(method == 'beam'):
        beam_width = st.slider("Beam Width", 2, 10, 3)
    if st.button("Clear History"):
        st.session_state.history = []

user_input = st.chat_input("Type your message...")

if user_input:
    sit = situation if situation else "general conversation"
    input_text = f"Emotion: {emotion} | Situation: {sit} | Customer: {user_input} Agent:"
    
    with st.spinner("Thinking..."):
        response = generate(model, input_text, vocab, inv_vocab, method=method, beam_width=beam_width)
    
    st.session_state.history.append(('user', user_input))
    st.session_state.history.append(('bot', response))

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)
