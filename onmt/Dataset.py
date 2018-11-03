from __future__ import division

import math
import torch
from torch.autograd import Variable

import onmt


class Dataset(object):
    """
    Manages dataset creation and usage.

    Example:

        `batch = data[batchnum]`
    """

    def __init__(self, srcData, tgtData, batchSize, cuda,
                 volatile=False, alignment=None):
        """
        Construct a data set

        Args:
            srcData, tgtData: The first parameter.
            batchSize: Training batchSize to use.
            cuda: Return batches on gpu.
            volitile:
            alignment: Alignment masks between src and tgt for copying.
        """
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
            self.tgtFeatures = None
        self.cuda = cuda
        self.alignment = alignment
        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile

    def _batchify(self, data, align_right=False,
                  include_lengths=False):

        lengths = [x.size(0) for x in data]
        max_length = max(lengths)

        out = data[0].new(len(data), max_length) \
                     .fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0

            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out


    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        s = index*self.batchSize
        e = (index+1)*self.batchSize
        batch_size = len(self.src[s:e])
        srcBatch, lengths = self._batchify(
            self.src[s:e],
            align_right=False, include_lengths=True)
        if srcBatch.dim() == 2:
            srcBatch = srcBatch.unsqueeze(2)
        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtBatch = None

        # Create a copying alignment.
        alignment = None
        if self.alignment:
            src_len = srcBatch.size(1)
            tgt_len = tgtBatch.size(1)
            batch = tgtBatch.size(0)
            alignment = torch.ByteTensor(tgt_len, batch, src_len).fill_(0)
            region = self.alignment[s:e]
            for i in range(len(region)):
                alignment[1:region[i].size(1)+1, i,
                          :region[i].size(0)] = region[i].t()
            alignment = alignment.float()

            if self.cuda:
                alignment = alignment.cuda()
        # tgt_len x batch x src_len
        lengths = torch.LongTensor(lengths)
        indices = range(len(srcBatch))
        # within batch sorting by decreasing length for variable length rnns
        lengths, perm = torch.sort(torch.LongTensor(lengths), 0,
                                   descending=True)
        indices = [indices[p] for p in perm]
        srcBatch = [srcBatch[p] for p in perm]
        if tgtBatch is not None:
            tgtBatch = [tgtBatch[p] for p in perm]
        if alignment is not None:
            alignment = alignment.transpose(0, 1)[
                perm.type_as(alignment).long()]
            alignment = alignment.transpose(0, 1).contiguous()

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0)
            b = b.transpose(0, 1).contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # Wrap lengths in a Variable to properly split it in DataParallel
        lengths = lengths.view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)

        return Batch(wrap(srcBatch),
                     wrap(tgtBatch),
                     lengths,
                     indices,
                     batch_size,
                     alignment=alignment)

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])


class Batch(object):
    """
    Object containing a single batch of data points.
    """
    def __init__(self, src, tgt, lengths, indices, batchSize, alignment=None):
        self.src = src
        self.tgt = tgt
        self.lengths = lengths
        self.indices = indices
        self.batchSize = batchSize
        self.alignment = alignment

    def words(self):
        return self.src[:, :, 0]

    def features(self, j):
        return self.src[:, :, j+1]

    def truncate(self, start, end):
        """
        Return a batch containing section from start:end.
        """
        return Batch(self.src, self.tgt[start:end],
                     self.lengths, self.indices, self.batchSize,
                     self.alignment[start:end])
