# -*- coding: utf-8 -*-
import torch
import onmt.Constants


def align(src_tokens, tgt_tokens):
    """
    Given two sequences of tokens, return
    a mask of where there is overlap.

    Returns:
        mask: src_len x tgt_len
    """
    mask = torch.ByteTensor(len(src_tokens), len(tgt_tokens)).fill_(0)

    for i in range(len(src_tokens)):
        for j in range(len(tgt_tokens)):
            if src_tokens[i] == tgt_tokens[j]:
                mask[i][j] = 1
    return mask


def readSrcLine(src_line, src_dict):
    srcWords= extractFeatures(src_line)
    srcData = src_dict.convertToIdx(srcWords,
                                    onmt.Constants.UNK_WORD)
    return srcWords, srcData


def readTgtLine(tgt_line, tgt_dict):
    tgtWords = extractFeatures(tgt_line)
    tgtData = tgt_dict.convertToIdx(tgtWords,
                                    onmt.Constants.UNK_WORD,
                                    onmt.Constants.BOS_WORD,
                                    onmt.Constants.EOS_WORD)

    return tgtWords, tgtData


def extractFeatures(tokens):
    "Given a list of token separate out words"
    words = []

    for t in range(len(tokens)):
        field = tokens[t].split(u"￨")
        word = field[0]
        if len(word) > 0:
            words.append(word)

    return words
