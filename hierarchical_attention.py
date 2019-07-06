import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import pickle


# Make the the multiple attention with word vectors.
def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


# The word RNN model for generating a sentence vector
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, batch_size, hid_size):
        super(WordRNN, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.hid_size = hid_size
        # Word Encoder
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.wordRNN = nn.GRU(embed_size, hid_size, bidirectional=True)
        # Word Attention
        self.wordattn = nn.Linear(2 * hid_size, 2 * hid_size)
        self.attn_combine = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)

    def forward(self, inp, hid_state):
        emb_out = self.embed(inp)

        out_state, hid_state = self.wordRNN(emb_out, hid_state)

        word_annotation = self.wordattn(out_state)
        attn = F.softmax(self.attn_combine(word_annotation), dim=1)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


# The sentence RNN model for generating a hunk vector
class SentRNN(nn.Module):
    def __init__(self, sent_size, hid_size):
        super(SentRNN, self).__init__()
        # Sentence Encoder
        self.sent_size = sent_size
        self.sentRNN = nn.GRU(sent_size, hid_size, bidirectional=True)

        # Sentence Attention
        self.sentattn = nn.Linear(2 * hid_size, 2 * hid_size)
        self.attn_combine = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.sentRNN(self.sent_size, hid_state)

        sent_annotation = self.sentattn(out_state)
        attn = F.softmax(self.attn_combine(sent_annotation), dim=1)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


# The HAN model
class HierachicalRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, batch_size, hid_size, c):
        super(HierachicalRNN, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.hid_size = hid_size
        self.cls = c

        # Word Encoder
        self.wordRNN = WordRNN(vocab_size, embed_size, batch_size, hid_size)
        # Sentence Encoder
        self.sentRNN = SentRNN(embed_size, hid_size)

        # Hunk Encoder
        self.sentRNN = nn.GRU(embed_size, hid_size, bidirectional=True)
        # Hunk Attention
        self.hunkattn = nn.Linear(2 * hid_size, 2 * hid_size)
        self.attn_combine = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)
        self.doc_linear = nn.Linear(2 * hid_size, c)

    def forward(self, inp, hid_state_sent, hid_state_word):
        s = None
        # Generating sentence vector through WordRNN
        for i in range(len(inp[0])):
            r = None
            for j in range(len(inp)):
                if r is None:
                    r = [inp[j][i]]
                else:
                    r.append(inp[j][i])
            r1 = np.asarray([sub_list + [0] * (self.max_seq_len - len(sub_list)) for sub_list in r])
            _s, state_word = self.wordRNN(torch.cuda.LongTensor(r1).view(-1, self.batch_size), hid_state_word)
            if s is None:
                s = _s
            else:
                s = torch.cat((s, _s), 0)

                out_state, hid_state = self.sentRNN(s, hid_state_sent)
        sent_annotation = self.sentattn(out_state)
        attn = F.softmax(self.attn_combine(sent_annotation), dim=1)

        doc = attention_mul(out_state, attn)
        d = self.doc_linear(doc)
        cls = F.log_softmax(d.view(-1, self.cls), dim=1)
        return cls, hid_state

