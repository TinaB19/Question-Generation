from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules
from onmt.modules import aeq
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class Embeddings(nn.Module):
    def __init__(self, opt, dicts):
        super(Embeddings, self).__init__()

        # vocab_sizes: sequence of vocab sizes for words and each feature
        vocab_sizes = [dicts.size()]
        # emb_sizes
        emb_sizes = [opt.word_vec_size]
        self.emb_luts = nn.ModuleList([
                                nn.Embedding(vocab, dim,
                                             padding_idx=onmt.Constants.PAD)
                                for vocab, dim in zip(vocab_sizes, emb_sizes)])

    @property
    def word_lut(self):
        return self.emb_luts[0]

    @property
    def embedding_size(self):
        return self.word_lut.embedding_dim

    def load_pretrained_vectors(self, emb_file):
        if emb_file is not None:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, src_input):
        """
        Embed the words or utilize features and MLP.

        Args:
            src_input (LongTensor): len x batch x nfeat

        Return:
            emb (FloatTensor): len x batch x emb_size
                                emb_size is word_vec_size 

        """
        in_length, in_batch, nfeat = src_input.size()
        aeq(nfeat, len(self.emb_luts))

        if len(self.emb_luts) == 1:
            emb = self.word_lut(src_input.squeeze(2))

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_length, out_length)
        aeq(emb_size, self.embedding_size)

        return emb


class Encoder(nn.Module):
    """
    Encoder recurrent neural network.
    """
    def __init__(self, opt, dicts):
        """
        Args:
            opt: Model options.
            dicts (`Dict`): The src dictionary
        """
        # Number of rnn layers.
        self.layers = opt.layers

        # Use a bidirectional model.
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0

        # Size of the encoder RNN.
        self.hidden_size = opt.rnn_size // self.num_directions

        super(Encoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts)

        input_size = self.embeddings.embedding_size

        self.rnn = getattr(nn, opt.rnn_type)(
             input_size, self.hidden_size,
             num_layers=opt.layers,
             dropout=opt.dropout,
             bidirectional=opt.brnn)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat
            lengths (LongTensor): batch
            hidden: Initial hidden state.

        Returns:
            hidden_t (FloatTensor): Pair of layers x batch x rnn_size - final
                                    Encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        # CHECKS
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            _, n_batch_ = lengths.size()
            aeq(n_batch, n_batch_)
        # END CHECKS

        emb = self.embeddings(input)
        s_len, n_batch, vec_size = emb.size()

        # Standard RNN encoder.
        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.data.view(-1).tolist()
            packed_emb = pack(emb, lengths)
        outputs, hidden_t = self.rnn(packed_emb, hidden)
        if lengths:
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class Decoder(nn.Module):
    """
    Decoder + Attention recurrent neural network.
    """

    def __init__(self, opt, dicts):
        """
        Args:
            opt: model options
            dicts: Target `Dict` object
        """
        self.layers = opt.layers
        self._coverage = opt.coverage_attn
        self.hidden_size = opt.rnn_size
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts)

        if opt.rnn_type == "LSTM":
            stackedCell = onmt.modules.StackedLSTM
        else:
            stackedCell = onmt.modules.StackedGRU
        self.rnn = stackedCell(opt.layers, input_size,
                               opt.rnn_size, opt.dropout)


        self.dropout = nn.Dropout(opt.dropout)

        # Std attention layer.
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size,
                                                 coverage=self._coverage,
                                                 attn_type=opt.attention_type)

    def forward(self, input, src, context, state):
        """
        Forward through the decoder.

        Args:
            input (LongTensor):  (len x batch) -- Input tokens
            src (LongTensor)
            context:  (src_len x batch x rnn_size)  -- Memory bank
            state: an object initializing the decoder.

        Returns:
            outputs: (len x batch x rnn_size)
            final_states: an object of the same form as above
            attns: Dictionary of (src_len x batch)
        """
        # CHECKS
        t_len, n_batch = input.size()
        s_len, n_batch_, _ = src.size()
        s_len_, n_batch__, _ = context.size()
        aeq(n_batch, n_batch_, n_batch__)
        # aeq(s_len, s_len_)
        # END CHECKS

        emb = self.embeddings(input.unsqueeze(2))

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []

        # Setup the different types of attention.
        attns = {"std": []}
        if self._coverage:
            attns["coverage"] = []

        assert isinstance(state, RNNDecoderState)
        output = state.input_feed.squeeze(0)
        hidden = state.hidden
        # CHECKS
        n_batch_, _ = output.size()
        aeq(n_batch, n_batch_)
        # END CHECKS

        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Standard RNN decoder.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(rnn_output,
                                          context.transpose(0, 1))

            output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # COVERAGE
            if self._coverage:
                coverage = (coverage + attn) if coverage else attn
                attns["coverage"] += [coverage]

        state = RNNDecoderState(hidden, output.unsqueeze(0),
                                coverage.unsqueeze(0)
                                if coverage is not None else None)
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])
        return outputs, state, attns


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim
        We need to convert it to layers x batch x (directions*dim)
        """
        if self.encoder.num_directions == 2:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        if isinstance(enc_hidden, tuple):
            dec = RNNDecoderState(tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:
            dec = RNNDecoderState(self._fix_enc_hidden(enc_hidden))
        dec.init_input_feed(context, self.decoder.hidden_size)
        return dec

    def forward(self, src, tgt, lengths, dec_state=None):
        """
        Args:
            src, tgt, lengths
            dec_state: A decoder state object

        Returns:
            outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_state = self.init_decoder_state(context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, src, context,
                                             enc_state if dec_state is None
                                             else dec_state)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    def detach(self):
        for h in self.all:
            if h is not None:
                h.detach_()

    def repeatBeam_(self, beamSize):
        self._resetAll([Variable(e.data.repeat(1, beamSize, 1))
                        for e in self.all])

    def beamUpdate_(self, idx, positions, beamSize):
        for e in self.all:
            a, br, d = e.size()
            sentStates = e.view(a, beamSize, br // beamSize, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, rnnstate, input_feed=None, coverage=None):
        # all objects are X x batch x dim
        # or X x (beam * sent) for beam search
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage
        self.all = self.hidden + (self.input_feed,)

    def init_input_feed(self, context, rnn_size):
        batch_size = context.size(1)
        h_size = (batch_size, rnn_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)
        self.all = self.hidden + (self.input_feed,)

    def _resetAll(self, all):
        vars = [Variable(a.data if isinstance(a, Variable) else a,
                         volatile=True) for a in all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
        self.all = self.hidden + (self.input_feed,)
