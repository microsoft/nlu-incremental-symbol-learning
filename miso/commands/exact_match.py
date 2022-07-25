# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import pickle as pkl 
from typing import List, Iterator, Dict  
from collections import namedtuple
import os
import overrides
import pdb 
import json

import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
import logging

from allennlp.commands.predict import _get_predictor, Predict
from allennlp.commands import ArgumentParserWithDefaults
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance
from allennlp.common.util import import_submodules

from miso.commands.predict import _CalFlowReturningPredictManager, _ReturningPredictManager
from miso.metrics.exact_match import BasicExactMatch, AdvancedExactMatch
from miso.metrics.fxn_metrics import SingleFunctionMetric, SyntheticFunctionMetric
from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from dataflow.core.lispress import render_compact, render_pretty, parse_lispress

logger = logging.getLogger(__name__) 

class ArgNamespace:
    def __init__(self,
                input_file,
                batch_size,
                silent,
                beam_size,
                oracle,
                top_k_beam_search,
                top_k):
        self.input_file = input_file
        self.batch_size = batch_size
        self.silent = silent
        self.beam_size = beam_size
        self.oracle = oracle
        self.top_k_beam_search = top_k_beam_search
        self.top_k = top_k


class ExactMatchScore(Subcommand): 
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        self.name = name 
        description = """Run the specified model against a JSON-lines input file."""
        subparser = parser.add_parser(
            self.name, description=description, help="Use a trained model to make predictions."
        )

        subparser.add_argument(
            "archive_file", type=str, help="the archived model to make predictions with"
        )
        subparser.add_argument("input_file", type=str, help="path to or url of the input file")

        subparser.add_argument("--output-file", type=str, help="path to output file")
        subparser.add_argument(
            "--weights-file", type=str, help="a path that overrides which weights file to use"
        )

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument(
            "--batch-size", type=int, default=1, help="The batch size to use for processing"
        )

        subparser.add_argument(
            "--silent", action="store_true", help="do not print output to stdout"
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "--use-dataset-reader",
            action="store_true",
            help="Whether to use the dataset reader of the original model to load Instances. "
            "The validation dataset reader will be used if it exists, otherwise it will "
            "fall back to the train dataset reader. This behavior can be overridden "
            "with the --dataset-reader-choice flag.",
        )

        subparser.add_argument(
            "--dataset-reader-choice",
            type=str,
            choices=["train", "validation"],
            default="validation",
            help="Indicates which model dataset reader to use if the --use-dataset-reader "
            "flag is set.",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )


        subparser.add_argument(
            "--predictor", type=str, help="optionally specify a specific predictor to use"
        )
        # my options
        subparser.add_argument('--beam-size',
                          type=int,
                          default=1,
                          help="Beam size for seq2seq decoding")

        subparser.add_argument('--out-file', type=str, default=None, help = "path to output tgt file")
        subparser.add_argument("--line-limit", type=int, default=None)
        subparser.add_argument("--score-type", choices=["advanced","basic"], default="advanced", help = "Type of exact match score to use. Advanced is the default and reference choice")
        subparser.add_argument("--fxn-of-interest", default=None, help = "Function to get coarse and fine-grained accuracy for")
        subparser.add_argument("--precomputed", action='store_true', required=False, help = "Don't run prediction again, just use already computed outputs ")
        subparser.add_argument("--oracle", action="store_true")
        subparser.add_argument("--top-k-beam-search", action="store_true", help="set to true if you want to decode the --top-k predictions instead of the top 1 from beam search")
        subparser.add_argument("--top-k", type=int, default=1, help = "top k to predict out of beam search") 
        subparser.add_argument("--json-save-path", type=str, help="if doing oracle decode, path to save instances and output")

        subparser.set_defaults(func=_construct_and_predict)

        return subparser



def _construct_and_predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)
    args.predictor = predictor
    scorer = Scorer.from_params(args)
    if args.oracle:
        # we're doing oracle things
        __, output = scorer.predict_and_compute()
        # save the output 
        with open(args.json_save_path, "w") as f1:
            json.dump(output, f1)
            print(f"Saved outputs to {args.json_save_path}")
        return

    if args.top_k_beam_search:
        __, output = scorer.predict_and_compute()
        return 

    if args.fxn_of_interest is not None:
        exact_match, coarse, fine, precision, recall, f1 = scorer.predict_and_compute()
        #(result, (coarse, fine)) = scorer.predict_and_compute()
    else:
        exact_match = scorer.predict_and_compute()

    print(f"Exact Match: {exact_match*100:.2f}") 

    if args.fxn_of_interest is not None:
        print(f"{args.fxn_of_interest} Coarse: {coarse*100:.2f}")
        print(f"{args.fxn_of_interest} Fine: {fine*100:.2f}")
        print(f"{args.fxn_of_interest} Precision: {precision*100:.2f}")
        print(f"{args.fxn_of_interest} Recall: {recall*100:.2f}")
        print(f"{args.fxn_of_interest} F1: {f1*100:.2f}")

class Scorer:
    """
    Reads, predicts, and scores graph structure using the exact match metric
    """
    def __init__(self,
                predictor = None,
                input_file = "dev",
                score_type = "advanced",
                batch_size = 32,
                silent = True,
                beam_size = 5, 
                line_limit = None,
                out_file = None,
                fxn_of_interest = None,
                oracle = False, 
                top_k_beam_search = False,
                top_k = 1,
                precomputed = False): 

        # otherwise we need to predict 
        self.predictor = predictor
        if predictor is not None:
            self.pred_args = ArgNamespace(input_file, batch_size, silent, beam_size, oracle=oracle, top_k_beam_search=top_k_beam_search, top_k=top_k) 

        self.line_limit = line_limit
        if score_type == "basic":
            self.metric = BasicExactMatch()
        else:
            self.metric = AdvancedExactMatch()

        self.fxn_of_interest = fxn_of_interest
        if fxn_of_interest is not None:
            if score_type == "basic":
                self.fxn_metric = SyntheticFunctionMetric(fxn_of_interest)
            else:
                self.fxn_metric = SingleFunctionMetric(fxn_of_interest)

        self.precomputed = precomputed
        self.oracle = oracle
        self.top_k_beam_search = top_k_beam_search
        self.top_k = top_k
        self.output_file = out_file 
        self.manager = _CalFlowReturningPredictManager(self.predictor,
                                    self.pred_args.input_file,
                                    None,
                                    self.pred_args.batch_size,
                                    not self.pred_args.silent,
                                    True,
                                    self.pred_args.beam_size,
                                    line_limit = self.line_limit,
                                    oracle = self.pred_args.oracle,
                                    top_k_beam_search=self.top_k_beam_search,
                                    top_k=self.top_k,
                                    precomputed = self.precomputed)

    @staticmethod
    def flatten_instance_batches(batch_iterator: Iterator[List[Instance]], 
                                total: int):
        flat = []
        for batch in tqdm(batch_iterator, total = total):
            for inst in batch:
                flat.append(inst.fields['calflow_graph'].metadata)
        return flat

    @staticmethod
    def flatten_prediction_batches(batch_iterator: Iterator[List[CalFlowGraph]], 
                                    total: int):
        flat = []
        for batch in tqdm(batch_iterator, total = total):
            for output in batch:
                flat.append(output)
        return flat
    
    @staticmethod
    def flatten_prediction_batch(batch):
        flat = []
        for result in batch:
            flat.append(result)
        return flat

    def predict_and_compute(self):
        assert(self.predictor is not None)
        input_instances, output_graphs = self.manager.run()

        if self.oracle:
            return input_instances, output_graphs

        if self.top_k_beam_search:
            with open(self.output_file, "w") as f1:
                for line in output_graphs:
                    lispress = parse_lispress(line)
                    string = render_compact(lispress)
                    f1.write(string.strip() + "\n")
            logger.info(f"Wrote {len(output_graphs)} hypotheses ({self.top_k} per input) to {self.output_file}")
            return None, None


        if self.precomputed:
            with open(self.output_file) as f1:
                output_lines = f1.readlines()
            output_graphs = [x.strip() for x in output_lines]

        try: 
            input_graphs = [inst.fields['calflow_graph'].metadata for inst in input_instances] 
            input_sents = [inst.fields['src_tokens_str'].metadata for inst in input_instances]
        except KeyError:
            # is a sequence not a graph
            input_graphs = [inst.fields['tgt_tokens_inputs'].metadata for inst in input_instances]
            input_sents = [inst.fields['src_tokens_str'].metadata for inst in input_instances]

        for inp, out in zip(input_graphs, output_graphs):
            if type(inp) == str and "Func" in inp and "(" not in inp:
                # dealing with synthetic data
                input_str = inp
            else:
                if type(inp) == str:
                    input_str = render_pretty(parse_lispress(inp))
                else:
                    input_str = render_pretty(inp.lispress)
            self.metric(input_str, out)
            if self.fxn_of_interest is not None:
                self.fxn_metric(input_str, out)

        if self.output_file is not None:
            with open(self.output_file, "w") as f1:
                for line in output_graphs:
                    if isinstance(self.metric, AdvancedExactMatch):
                        lispress = parse_lispress(line)
                        string = render_compact(lispress)
                    else:
                        # synthetic data 
                        string = line
                    f1.write(string.strip() + "\n")

        to_ret = [self.metric.get_metric(reset=True)]
        if self.fxn_of_interest is not None:
            to_ret += list(self.fxn_metric.get_metric(reset=True))
        return to_ret
   
 
    @classmethod
    def from_params(cls, args):
        return cls(predictor=args.predictor,
                   input_file = args.input_file,
                   batch_size = args.batch_size,
                   score_type = args.score_type, 
                   silent = args.silent,
                   beam_size = args.beam_size, 
                   line_limit = args.line_limit,
                   out_file = args.out_file,
                   fxn_of_interest = args.fxn_of_interest,
                   precomputed = args.precomputed,
                   oracle=args.oracle,
                   top_k_beam_search=args.top_k_beam_search,
                   top_k=args.top_k
                   )

if __name__ == "__main__":
    parser = ArgumentParserWithDefaults(description="Run AllenNLP")
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "eval": ExactMatchScore(),
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        # configure doesn't need include-package because it imports
        # whatever classes it needs.
        if name != "configure":
            subparser.add_argument('--include-package',
                                   type=str,
                                   action='append',
                                   default=[],
                                   help='additional packages to include')

    args = parser.parse_args()
    if 'func' in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, 'include_package', ()):
            import_submodules(package_name)
        args.func(args)

