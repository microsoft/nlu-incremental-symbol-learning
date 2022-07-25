# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np   
import pdb 
import argparse
import pathlib 
np.random.seed(12)



INPUT_OUTPUT={"a": "Func1", "b": "Func2", "c": "Func3"}

def write_splits(train, dev, test, path):
    with open(path.joinpath("train.src_tok"),"w") as train_src, open(path.joinpath("train.tgt"), "w") as train_tgt:
        for ts, tt in train:
            train_src.write(ts.strip() + "\n")
            train_tgt.write(tt.strip() + "\n")
    with open(path.joinpath("dev.src_tok"),"w") as dev_src, open(path.joinpath("dev.tgt"), "w") as dev_tgt:
        for ds, dt in dev:
            dev_src.write(ds.strip() + "\n")
            dev_tgt.write(dt.strip() + "\n")
    with open(path.joinpath("test.src_tok"),"w") as test_src, open(path.joinpath("test.tgt"), "w") as test_tgt:
        for tss, tst in test:
            test_src.write(tss.strip() + "\n")
            test_tgt.write(tst.strip() + "\n")

def generate_single(out_path):
    out_path = out_path.joinpath("single_output")
    for num in [1, 5, 10, 20, 50]:
        for split in [50, 500, 1000, 2000, 5000, 10000, 12000]:
            num_a = split - num
            train_a_src = ["a" for __ in range(num_a)]
            train_a_tgt = [INPUT_OUTPUT[a] for a in train_a_src]
            train_b_src = ["b" for __ in range(num)]
            train_b_tgt = [INPUT_OUTPUT[b] for b in train_b_src]

            train_src = train_a_src + train_b_src
            train_tgt = train_a_tgt + train_b_tgt
            train_data = [x for x in zip(train_src, train_tgt)]
            np.random.shuffle(train_data)

            # make balanced testdev 
            testdev_a_src = ["a" for __ in range(2700)]
            testdev_a_tgt = [INPUT_OUTPUT[a] for a in testdev_a_src]
            testdev_b_src = ["b" for __ in range(2700)]
            testdev_b_tgt = [INPUT_OUTPUT[b] for b in testdev_b_src]


            td_len = len(testdev_a_src)
            half_td_len = int(td_len/2)

            test_a_src, test_a_tgt = testdev_a_src[0:half_td_len], testdev_a_tgt[0:half_td_len]
            test_b_src, test_b_tgt = testdev_b_src[0:half_td_len], testdev_b_tgt[0:half_td_len]
            test_src = test_a_src + test_b_src
            test_tgt = test_a_tgt + test_b_tgt
            test_data = [x for x in zip(test_src, test_tgt)]
            np.random.shuffle(test_data)

            dev_a_src, dev_a_tgt = testdev_a_src[half_td_len:], testdev_a_tgt[half_td_len:]
            dev_b_src, dev_b_tgt = testdev_b_src[half_td_len:], testdev_b_tgt[half_td_len:]
            dev_src = dev_a_src + dev_b_src
            dev_tgt = dev_a_tgt + dev_b_tgt
            dev_data = [x for x in zip(dev_src, dev_tgt)]
            np.random.shuffle(dev_data)

            final_path = out_path.joinpath(f"Func2", f"{split}_{num}")
            final_path.mkdir(parents=True, exist_ok=True)
            write_splits(train_data, dev_data, test_data, final_path)

def make_disjoint(train_b_src, train_b_tgt, testdev_a_src, testdev_a_tgt, testdev_b_src, testdev_b_tgt):
    train_src_types = set(train_b_src)
    train_tgt_types = set(train_b_tgt)

    new_b_src, new_b_tgt = [], []
    new_a_src, new_a_tgt = [], []
    for b_src, b_tgt in zip(testdev_b_src, testdev_b_tgt):
        # don't add anything that's in train
        if b_src in train_src_types:
            continue
        # don't add anything that's already added
        if b_src in new_b_src:
            continue
        new_b_src.append(b_src)
        new_b_tgt.append(b_tgt)
        # add the same length a example 
        new_a_src.append(" ".join(["a" for i in range(len(b_src.split(" ")))]))
        new_a_tgt.append(" ".join(["Func1" for i in range(len(b_tgt.split(" ")))]))

    return new_a_src, new_a_tgt, new_b_src, new_b_tgt 

def make_seq2seq_data(split, num, out_path, max_len, three_piece, b_options):
    out_path = out_path.joinpath("seq2seq")
    def generate_b():
        # max 1 b per string 
        src = ["a" for i in range(np.random.randint(low=1, high=max_len))]
        b_idx = np.random.choice([i for i in range(len(src))])
        src[b_idx] = 'b'
        tgt = [INPUT_OUTPUT[c] for c in src]
        return " ".join(src), " ".join(tgt)

    def generate_a(char='a'):
        src = [char for i in range(np.random.randint(low=1, high=max_len))]
        tgt = [INPUT_OUTPUT[c] for c in src]
        return " ".join(src), " ".join(tgt)
    num_a = split - num
    train_a_src, train_a_tgt = zip(*[generate_a() for __ in range(num_a)])
    if three_piece:
        train_c_src, train_c_tgt = zip(*[generate_a(char="c") for __ in range(num_a)])

    # don't generate, do this so that they all have the same b sentences for every split 
    train_b_src, train_b_tgt = zip(*b_options[0:num])

    #train_b_src, train_b_tgt = zip(*[generate_b() for __ in range(num)])

    train_src = train_a_src + train_b_src
    train_tgt = train_a_tgt + train_b_tgt
    if three_piece:
        train_src += train_c_src
        train_tgt += train_c_tgt
    train_data = [x for x in zip(train_src, train_tgt)]
    np.random.shuffle(train_data)

    # make balanced testdev 
    testdev_a_src, testdev_a_tgt = zip(*[generate_a() for __ in range(2700)])
    if three_piece:
        testdev_c_src, testdev_c_tgt = zip(*[generate_a(char='c') for __ in range(2700)])
    testdev_b_src, testdev_b_tgt = zip(*[generate_b() for __ in range(2700)])

    # make disjoint 
    testdev_a_src, testdev_a_tgt, testdev_b_src, testdev_b_tgt = make_disjoint(train_src, train_tgt, testdev_a_src, testdev_a_tgt, testdev_b_src, testdev_b_tgt )

    td_len = len(testdev_a_src)
    half_td_len = int(td_len/2)


    test_a_src, test_a_tgt = testdev_a_src[0:half_td_len], testdev_a_tgt[0:half_td_len]
    test_b_src, test_b_tgt = testdev_b_src[0:half_td_len], testdev_b_tgt[0:half_td_len]
    if three_piece:
        test_c_src, test_c_tgt = testdev_c_src[0:half_td_len], testdev_c_tgt[0:half_td_len]

    test_src = test_a_src + test_b_src
    test_tgt = test_a_tgt + test_b_tgt
    if three_piece:
        test_src += test_c_src
        test_tgt += test_c_tgt

    test_data = [x for x in zip(test_src, test_tgt)]
    np.random.shuffle(test_data)

    dev_a_src, dev_a_tgt = testdev_a_src[half_td_len:], testdev_a_tgt[half_td_len:]
    dev_b_src, dev_b_tgt = testdev_b_src[half_td_len:], testdev_b_tgt[half_td_len:]
    if three_piece:
        dev_c_src, dev_c_tgt = testdev_c_src[half_td_len:], testdev_c_tgt[half_td_len:]

    dev_src = dev_a_src + dev_b_src
    dev_tgt = dev_a_tgt + dev_b_tgt
    if three_piece:
        dev_src += dev_c_src
        dev_tgt += dev_c_tgt 
    dev_data = [x for x in zip(dev_src, dev_tgt)]
    np.random.shuffle(dev_data)

    final_path = out_path.joinpath(f"Func2", f"{split}_{num}")
    final_path.mkdir(parents=True, exist_ok=True)
    write_splits(train_data, dev_data, test_data, final_path)

def get_b_options(max_len):
    b_opts = []
    for i in range(1, max_len):
        base_src = ["a" for __ in range(i)]
        base_tgt = ["Func1" for __ in range(i)]

        for j in range(len(base_src)):
            new_base_src = [x for x in base_src]
            new_base_tgt = [x for x in base_tgt]
            new_base_src[j] = "b"
            new_base_tgt[j] = "Func2"
            b_opts.append((" ".join(new_base_src), " ".join(new_base_tgt)))

    np.random.shuffle(b_opts)
    return b_opts

def generate_seq2seq(out_path, max_len, three_piece=False):
    b_options = get_b_options(max_len)


    for num in [1, 5, 10, 20, 50]:
        for split in [100, 500, 1000, 2000, 5000, 10000, 12000]:
            make_seq2seq_data(split, num, out_path, max_len, three_piece, b_options)

    # make 50-50 data
    for split in [100, 500, 1000, 2000, 5000, 10000, 12000]:
        make_seq2seq_data(split, int(split/2), out_path, max_len, three_piece, b_options)

def main(args):
    out_path = pathlib.Path(args.out_path)
    # generate single input single output 
    generate_single(out_path)
    # generate seq2seq 
    generate_seq2seq(out_path, args.max_len, args.three_piece)
    # generate multi-label classification 
    # generate_multi_class(args.out_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--three-piece", action="store_true", default=False, required=False) 
    args = parser.parse_args() 

    main(args)
