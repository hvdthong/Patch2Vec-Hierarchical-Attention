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
    def __init__(self, vocab_size, embed_size, batch_size, hidden_size):
        super(WordRNN, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # Word Encoder
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.wordRNN = nn.GRU(embed_size, hidden_size, bidirectional=True)
        # Word Attention
        self.wordattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        emb_out = self.embed(inp)

        out_state, hid_state = self.wordRNN(emb_out, hid_state)

        word_annotation = self.wordattn(out_state)
        attn = F.softmax(self.attn_combine(word_annotation), dim=1)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


# The sentence RNN model for generating a hunk vector
class SentRNN(nn.Module):
    def __init__(self, sent_size, hidden_size):
        super(SentRNN, self).__init__()
        # Sentence Encoder
        self.sent_size = sent_size
        self.sentRNN = nn.GRU(sent_size, hidden_size, bidirectional=True)

        # Sentence Attention
        self.sentattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.sentRNN(inp, hid_state)

        sent_annotation = self.sentattn(out_state)
        attn = F.softmax(self.attn_combine(sent_annotation), dim=1)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


# The hunk RNN model for generating the vector representation for the instance
class HunkRNN(nn.Module):
    def __init__(self, sent_size, hidden_size):
        super(HunkRNN, self).__init__()
        # Sentence Encoder
        self.sent_size = sent_size
        self.hunkRNN = nn.GRU(sent_size, hidden_size, bidirectional=True)

        # Sentence Attention
        self.hunkattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.hunkRNN(inp, hid_state)

        hunk_annotation = self.hunkattn(out_state)
        attn = F.softmax(self.attn_combine(hunk_annotation), dim=1)

        hunk = attention_mul(out_state, attn)
        return hunk, hid_state


# The HAN model
class HierachicalRNN(nn.Module):
    def __init__(self, args):
        super(HierachicalRNN, self).__init__()
        self.vocab_size = args.vocab_code
        self.batch_size = args.batch_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.cls = args.class_num

        # Word Encoder
        self.wordRNN = WordRNN(self.vocab_size, self.embed_size, self.batch_size, self.hidden_size)
        # Sentence Encoder
        self.sentRNN = SentRNN(self.embed_size, self.hidden_size)
        # Hunk Encoder
        self.hunkRNN = HunkRNN(self.embed_size, self.hidden_size)

        # Hidden layers before putting to the output layer
        self.doc_linear = nn.Linear(2 * self.hidden_size, self.cls)

    def forward_code(self, x, hid_state):
        hid_state_hunk, hid_state_sent, hid_state_word = hid_state
        n_batch, n_hunk, n_line = x.shape[0], x.shape[1], x.shape[2]
        # i: hunk; j: line; k: batch
        hunks = list()
        for i in range(n_hunk):
            sents = None
            for j in range(n_line):
                words = list()
                for k in range(n_batch):
                    words.append(x[k][i][j])
                words = np.array(words)
                sent, state_word = self.wordRNN(torch.cuda.LongTensor(words).view(-1, self.batch_size), hid_state_word)
                if sents is None:
                    sents = sent
                else:
                    sents = torch.cat((sents, sent), 0)
            hunk, state_sent = self.sentRNN(sents, hid_state_sent)
            if hunks is None:
                hunks = hunk
            else:
                hunks = torch.cat((hunks, hunk), 0)
        out_hunk, state_hunk = self.hunkRNN(hunks, hid_state_hunk)
        return out_hunk

    def forward(self, added_code, removed_code, hid_state_hunk, hid_state_sent, hid_state_word):
        hid_state = (hid_state_hunk, hid_state_sent, hid_state_word)
        x_added_code = self.forward_code(x=added_code, hid_state=hid_state)
        return None

    def init_hidden_hunk(self):
        return Variable(torch.zeros(2, self.batch_size, self.hidden_size)).cuda()

    def init_hidden_sent(self):
        return Variable(torch.zeros(2, self.batch_size, self.hidden_size)).cuda()

    def init_hidden_word(self):
        return Variable(torch.zeros(2, self.batch_size, self.hidden_size)).cuda()