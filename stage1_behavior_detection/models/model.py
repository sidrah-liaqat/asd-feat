# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
from __future__ import print_function, division

from torch import optim, nn
import torch
import torch.nn.functional as F
import math

# naming the label as action because it sounds cooler
def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """
    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        self.k = k
        self.t = seq_length
        self.num_tokens = num_tokens
        # self.token_emb = nn.Embedding(num_tokens, k)
        # here words are replaced by landmarks,
        # so replacing embedding with linear layer
        self.headpose = nn.Linear(11, 6)
        self.landmarks = nn.Linear(136, 30)
        self.eyelandmarks = nn.Linear(112, 30)
        self.au = nn.Linear(35, 10)
        self.rgb = nn.Linear(64, 16)
        self.flow = nn.Linear(64, 16)
        # k = 30+10=40
        self.pos_emb = nn.Embedding(seq_length, k)  # k

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        # self.toprobs = nn.Linear(k, num_classes)
        self.toprobs = nn.Linear(k, 1)
        # self.constant_value = 0.3  # Replace with your desired constant
        # self.bias_constant = torch.tensor(self.constant_value).repeat(1)

        # Maps the final output sequence to regression distances
        self.reghead = nn.Linear(k, 2)

    def forward(self, dev, feat_head, feat_lmk, feat_eye, feat_au, rgb, flow):
        # def forward(self, dev, feat_head, feat_lmk, feat_eye, feat_au):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # v.shape = (b, frame, 6) eyegaze also added along with headpose
        # x.shape = (b, frame, 137)
        # y.shape = (b, frame, 113)
        # z.shape = (b, frame, 35)
        # rgb.shape = (b, 1024) --> reshaped to (b, frame, 16)
        # flow.shape = (b, 1024) --> reshaped to (b, frame, 16)

        tokens0 = self.headpose(feat_head)
        tokens1 = self.landmarks(feat_lmk)
        tokens2 = self.eyelandmarks(feat_eye)
        tokens3 = self.au(feat_au)
        # last two dimensions 16, 64 for cnns
        tokensrgb = self.rgb(rgb.view(tokens3.shape[0], 16, 64))
        tokensflow = self.flow(flow.view(tokens3.shape[0], 16, 64))

        tokens = torch.cat([tokens0, tokens1, tokens2, tokens3, tokensrgb, tokensflow], dim=2)

        # create new torch tensor with same size as 'tokens' and filled with zeros
        # b, t, k = tokens.size()
        b = tokens.shape[0]

        # generate position embeddings
        positions = torch.arange(self.t)

        # self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        positions = positions.to(dev)
        positions = self.pos_emb(positions)[None, :, :].expand(b, self.t, self.k)
        # print(positions.shape)
        # print(tokens.shape)
        if (tokens.shape != positions.shape):
            print("pause")
        positions_zero = torch.zeros_like(positions, dtype=torch.float32, device=dev)
        tokens_zero = torch.zeros_like(tokens, dtype=torch.float32, device=dev)

        outp = tokens + positions  # ?? mathematical addition??
        outp = self.tblocks(outp)

        # Average-pool over the t dimension and project to class
        # probabilities
        # outp = self.toprobs(outp.mean(dim=1))
        outcls = self.toprobs(outp)

        return outcls
        # return outcls

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8, mask=False):
        """
        :param k:
        :param heads:
        :param mask:
        """
        super().__init__()

        self.k = k
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)

        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):

        b, t, e = x.size()

        h = self.heads
        assert e == self.k, f'Input embedding dim ({e}) should match layer embedding dim ({self.k})'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(e)  # dot contains b*h  t-by-t matrices with raw self-attention logits

        assert dot.size() == (b * h, t, t), f'Matrix has size {dot.size()}, expected {(b * h, t, t)}.'

        if self.mask:  # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)  # dot now has row-wise self-attention probabilities

        # assert not util.contains_nan(dot[:, 1:, :]) # only the first row may contain nan
        assert not contains_nan(dot[:, 1:, :])  # only the first row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        return x

import torch.nn.init as init  # Import for weight initialization

class MLP_I3D(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Define layers
        self.layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 16),
        )

        # Initialize weights (example using Xavier Normal initialization)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)

    def forward(self, x, y):
        # Pass input through layers sequentially
        return self.layers(torch.cat([x, y], dim=1))  # (batch_size, 16x)