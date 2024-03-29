# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import os
import io
import sys
import argparse
import torch
import numpy as np

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel

from src.fp16 import network_to_half


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--fp16", type=bool_flag, default=False, help="Run model with float16")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    
    parser.add_argument("--beam", type=int, default=1, help="Beam size")
    parser.add_argument("--length_penalty", type=float, default=1, help="length penalty")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    return parser


def encoderouts(encoder, batch, lengths, langs):
    encoder.eval()
    # Run encoder
    encodings = encoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
    # -> deal with padding
    L, bsz = batch.size()
    # Create a mask with 0s where padding tokens are located
    positions = torch.arange(L).view(-1, 1).repeat(1, bsz) #.float()
    mask = positions.to(encodings.device).lt(lengths.cuda().view(1, -1))
    # Multiply by mask and sum over length dimension
    mean_encodings = torch.einsum("lb,lbd->bd", [mask.float(), encodings])
    # Normalize by respective lengths
    mean_encodings /= lengths.float().cuda().view(-1, 1)
    # Detach and return
    return mean_encodings.detach()
    

def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()
    encoder.load_state_dict(reloaded['encoder'])
    decoder = None
#    decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()
#    decoder.load_state_dict(reloaded['decoder'])
    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]


    # read sentences from stdin
    src_sent = []
    for line in sys.stdin.readlines():
        assert len(line.strip().split()) > 0
        src_sent.append(line)
    logger.info("Read %i sentences from stdin. Translating ..." % len(src_sent))


    all_encodings = []
    # For each sentence...
    for i in range(0, len(src_sent), params.batch_size):
        # prepare batch
        word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                    for s in src_sent[i:i + params.batch_size]]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
        batch[0] = params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = params.eos_index
        langs = batch.clone().fill_(params.src_id)

        # encode source batch and translate it, deal with padding
        encodings = encoderouts(encoder, batch, lengths, langs)
        
        # batch is actually in original order, append each sent to all_encodings
        for idx in encodings:
            all_encodings.append(idx.cpu().numpy())

    # Save all encodings to npy
    np.save(params.output_path, np.stack(all_encodings))


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang
    assert params.output_path and not os.path.isfile(params.output_path)

    # translate
    with torch.no_grad():
        main(params)
