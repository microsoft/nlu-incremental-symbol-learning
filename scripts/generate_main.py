# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import pdb 
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

from harbor_ext.cold_fusion import scfg
from harbor_ext.cold_fusion.generate import generate_synchronously
from harbor_ext.cold_fusion.read_grammar import PreprocessedGrammar
from harbor_ext.cold_fusion.scfg import SCFG
from dataflow.core.lispress import parse_lispress
from dataflow.core.linearize import lispress_to_seq

np.random.seed(12)

def generate_nonce(n, use_fn=True):
    if use_fn:
        return [f"Func{i}" for i in range(n)]
    else:
        #template = "{c1}{v}{c2}"
        consonants = [char for char in 'bcdfghjklmnpqrstvwxyz']
        vowels = [char for char in "aeiou"]
        # enforce these just for fun
        words = ["dax", "wug"]
        for i in range(n-2):
            cons = np.random.choice(consonants, size=2)
            c1, c2 = cons[0], cons[1]
            v = np.random.choice(vowels, size=1)[0]
            word = f"{c1}{v}{c2}"
            words.append(word)        
    return words

def generate_mapping(words, max_len, num_chars = 62):
    all_chars = [char for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789']
    input_symbols = all_chars[0:num_chars]
    input_words = []
    for i in range(0, len(input_symbols), max_len):
        for j in range(max_len):
            input_words.append(" ".join(input_symbols[i:i+j+1]))

    mapping = {}
    for word in input_words:
        mapping[word] = np.random.choice(words)

    return mapping 

def write_function_grammar(mapping, max_depth, scfg_dir):
    with open(pathlib.Path(scfg_dir).joinpath("functions.scfg"),"w") as f1:
        for input_word, output_word in mapping.items():
            line = f'function -> "{input_word}", "{output_word}"\n'
            f1.write(line)

    with open(pathlib.Path(scfg_dir).joinpath("sentences.scfg"), "w") as f1:
        last = 0
        for i in range(max_depth):
            line0 = f'sentence{i+1} -> sentence_end, sentence_end\n'
            line1 = f'sentence{i+1} -> sentence_end sentence{i+2},  sentence_end " " sentence{i+2}\n'
            f1.write(line0)
            f1.write(line1)
            last = i
        final_line = f'sentence{last+2} -> sentence_end, sentence_end\n'
        f1.write(final_line)

def write_full(args):
    words = generate_nonce(args.num_nonce)
    # super simple 1-to-1 mapping 
    mapping = generate_mapping(words, max_len=args.max_len, num_chars=args.num_chars)

    write_function_grammar(mapping, args.max_depth, args.scfg)

    grammar = SCFG(PreprocessedGrammar.from_folder(args.scfg))

    data = generate_synchronously(grammar, randomize=True)
    inputs = []
    outputs = []
    keep_generating = True
    while len(inputs) < args.num_examples and keep_generating:
        utterance_node, plan_node = next(data)

        seaweed = plan_node.render_topological()
        text=utterance_node.render()

        inputs.append(text)
        outputs.append(seaweed)

        # with p = 0.50, switch it up so data isn't all the same function 
        switch = np.random.randint(0, high=100)
        if switch < 50:
            data = generate_synchronously(grammar, randomize=True)


        if (len(inputs)+1) % 1000 == 0:
            print(len(inputs))
            print(seaweed)
            print(text)

    out_path = pathlib.Path(args.out_path)
    with open(out_path.joinpath("synthetic_full.src_tok"), "w") as src, open(out_path.joinpath("synthetic_full.tgt"), "w") as tgt, open(out_path.joinpath("synthetic_full.datum_id"), "w") as ids:
        for i, (input, output) in enumerate(zip(inputs, outputs)):
            #output = " ".join(lispress_to_seq(parse_lispress(output)))
            src.write(f"{input}\n")
            tgt.write(f"{output}\n")
            ids.write(f"{i}\n")

def split_train_dev_test(out_path):
    out_path = pathlib.Path(args.out_path)
    with open(out_path.joinpath("synthetic_full.src_tok")) as src, open(out_path.joinpath("synthetic_full.tgt")) as tgt, open(out_path.joinpath("synthetic_full.datum_id")) as ids:
        src_data = [x.strip() for x in src.readlines()]
        tgt_data = [x.strip() for x in tgt.readlines()]
        ids_data = [x.strip() for x in ids.readlines()]

    all_data = [x for x in zip(src_data, tgt_data, ids_data)]
    np.random.shuffle(all_data)
    train_start = 0
    train_end = int(0.7 * len(all_data))
    dev_start = train_end
    dev_end = int(0.85 * len(all_data))
    test_start = dev_end 

    train_data = all_data[train_start:train_end]
    dev_data = all_data[dev_start:dev_end]
    test_data = all_data[test_start:]

    with open(out_path.joinpath("train.src_tok"), "w") as src, open(out_path.joinpath("train.tgt"), "w") as tgt, open(out_path.joinpath("train.datum_id"), "w") as ids:
        for input, output, dat_id in train_data:
            src.write(f"{input}\n")
            tgt.write(f"{output}\n")
            ids.write(f"{dat_id}\n")

    with open(out_path.joinpath("dev.src_tok"), "w") as src, open(out_path.joinpath("dev.tgt"), "w") as tgt, open(out_path.joinpath("dev.datum_id"), "w") as ids:
        for input, output, dat_id in dev_data:
            src.write(f"{input}\n")
            tgt.write(f"{output}\n")
            ids.write(f"{dat_id}\n")

    with open(out_path.joinpath("test.src_tok"), "w") as src, open(out_path.joinpath("test.tgt"), "w") as tgt, open(out_path.joinpath("test.datum_id"), "w") as ids:
        for input, output, dat_id in test_data:
            src.write(f"{input}\n")
            tgt.write(f"{output}\n")
            ids.write(f"{dat_id}\n")

def main(args):
    write_full(args)
    split_train_dev_test(args.out_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scfg", default=None, required=True, help="path to .scfg dir") 
    parser.add_argument("--out-path", default=None, required=True, help="path to output dir") 
    parser.add_argument("--num-examples", default=100000, type=int) 
    parser.add_argument("--num-nonce", type=int, default=100, help="number of nonce words to use")
    parser.add_argument("--max-depth", type=int, default=10, help="max depth of function tree")
    parser.add_argument("--max-len", type=int, default=3, help="max len of input symbols")
    parser.add_argument("--num-chars", type=int, default=62, help="number of input character types")
    args = parser.parse_args()
    main(args)

