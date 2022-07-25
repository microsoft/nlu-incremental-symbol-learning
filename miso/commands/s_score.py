# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import pickle as pkl 
from typing import List, Iterator, Dict  
from collections import namedtuple
import os
import overrides
import pdb 

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

from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph
from miso.metrics.s_metric.s_metric import S, TEST1, NORMAL, compute_s_metric
from miso.metrics.s_metric.repr import Triple, FloatTriple
from miso.metrics.s_metric import utils
from miso.commands.predict import _ReturningPredictManager 
from miso.commands.conllu_score import ConlluScore
from miso.commands.conllu_predict import ConlluPredict 

logger = logging.getLogger(__name__) 

compute_args = {"seed": 0,
                "iter_num": 1, 
                "compute_instance": True,
                "compute_attribute": True,
                "compute_relation": True,
                "log_level": None,
                "sanity_check": False,
                "mode": NORMAL
                }

ComputeTup = namedtuple("compute_args", sorted(compute_args))

class ArgNamespace:
    def __init__(self,
                input_file,
                batch_size,
                silent,
                beam_size,
                save_pred_path):
        self.input_file = input_file
        self.batch_size = batch_size
        self.silent = silent
        self.beam_size = beam_size
        self.save_pred_path = save_pred_path


class SScore(Subcommand): 
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

        subparser.add_argument("--save-pred-path", type=str, required=False, 
                                help="optionally specify a path for output pkl") 

        subparser.add_argument("--load-path", type=str, required=False, 
                                help="path to precomuted predicitons") 
        
        subparser.add_argument("--semantics-only", action="store_true", default=False)

        subparser.add_argument("--drop-syntax", action="store_true", default=False)

        subparser.add_argument("--include-attribute-scores", action="store_true", default=False)

        subparser.add_argument("--line-limit", type=int, default=None)

        subparser.add_argument("--json-output-file", type=str, required=False,
                                help="optionally specify a path to output json dict") 

        subparser.add_argument("--oracle", action = "store_true") 

        subparser.set_defaults(func=_construct_and_predict)

        return subparser

def _construct_and_predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)
    args.predictor = predictor
    scorer = Scorer.from_params(args)
    if args.oracle:
        scorer.predict_and_save_oracle()
    else:
        p, r, f1 = scorer.predict_and_compute()
        print(f"Precision: {p}, Recall: {r}, F1: {f1}") 

class Scorer:
    """
    Reads, predicts, and scores graph structure using the S metric 
    """
    def __init__(self,
                predictor = None,
                input_file = "dev",
                batch_size = 32,
                silent = True,
                beam_size = 5, 
                save_pred_path = None,
                load_path = None,
                semantics_only = False,
                drop_syntax = True,
                line_limit = None,
                include_attribute_scores = False,
                oracle = False,
                json_output_file = None):

        self.load_path = load_path
        if self.load_path is not None:
            # don't use a predictor if provided with pre-computed graphs
            pass
        else:
            # otherwise we need to predict 
            self.predictor = predictor
            if predictor is not None:
                self.pred_args = ArgNamespace(input_file, batch_size, silent, beam_size, save_pred_path)

        self.semantics_only = semantics_only
        self.drop_syntax = drop_syntax
        self.line_limit = line_limit
        self.include_attribute_scores = include_attribute_scores
        self.oracle = oracle
        self.json_output_file = json_output_file

        
        self.manager = _ReturningPredictManager(self.predictor,
                                    self.pred_args.input_file,
                                    None,
                                    self.pred_args.batch_size,
                                    not self.pred_args.silent,
                                    True,
                                    self.pred_args.beam_size,
                                    line_limit = self.line_limit,
                                    oracle = self.oracle,
                                    json_output_file = self.json_output_file)

    @staticmethod
    def flatten_instance_batches(batch_iterator: Iterator[List[Instance]], 
                                total: int):
        flat = []
        for batch in tqdm(batch_iterator, total = total):
            for inst in batch:
                flat.append(inst.fields['graph'].metadata)
        return flat

    @staticmethod
    def flatten_prediction_batches(batch_iterator: Iterator[List[DecompGraph]], 
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

    def load_graphs(self, path: str):
        with open(path, 'rb') as f1: 
            true_graphs, pred_graphs = zip(*pkl.load(f1))
        # deserialize
        return [nx.adjacency_graph(x) for x in true_graphs], [nx.adjacency_graph(x) for x in pred_graphs]

    def save_graphs(self, 
                         true_graphs: List[DecompGraph],
                         pred_graphs: List[DecompGraph],
                         path: str):
        with open(path, "wb") as f1:
            pkl.dump(list(zip(true_graphs, pred_graphs)), f1)

    def predict_and_compute(self):
        assert(self.predictor is not None)
        input_instances, output_graphs = self.manager.run()

        if len(output_graphs) > 0 and type(output_graphs[0]) == tuple:
            # ignore conllu graphs here 
            output_graphs = [x[0] for x in output_graphs]

        input_graphs = [inst.fields['graph'].metadata for inst in input_instances] 
        input_sents = [inst.fields['src_tokens_str'].metadata for inst in input_instances]

        if self.pred_args.save_pred_path is not None:
            # save serialized graphs to pkl file  
            try:
                self.save_graphs([nx.adjacency_data(x) for x in input_graphs],
                                 [nx.adjacency_data(x) for x in output_graphs], 
                                self.pred_args.save_pred_path)
            except AttributeError:
                self.save_graphs([nx.adjacency_data(x) for x in input_graphs],
                                 [nx.adjacency_data(x[0]) for x in output_graphs], 
                                self.pred_args.save_pred_path)
                
        
        return compute_s_metric(input_graphs, 
                                output_graphs, 
                                input_sents, 
                                semantics_only = self.semantics_only,
                                drop_syntax = self.drop_syntax,
                                #args, 
                                include_attribute_scores = self.include_attribute_scores)
   
    def predict_and_save_oracle(self):
        assert(self.predictor is not None)
        input_instances, output_graphs = self.manager.run()

        return 
 
    @classmethod
    def from_params(cls, args):
        return cls(predictor=args.predictor,
                   input_file = args.input_file,
                   batch_size = args.batch_size,
                   silent = args.silent,
                   beam_size = args.beam_size, 
                   save_pred_path = args.save_pred_path,
                   load_path = args.load_path,
                   semantics_only = args.semantics_only,
                   drop_syntax = args.drop_syntax,
                   line_limit = args.line_limit,
                   include_attribute_scores = args.include_attribute_scores,
                   oracle = args.oracle,
                   json_output_file = args.json_output_file
                   )

if __name__ == "__main__":
    parser = ArgumentParserWithDefaults(description="Run AllenNLP")
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "eval": SScore(),
            "spr_eval": SScore(),
            "conllu_eval": ConlluScore(),
            "conllu_predict": ConlluPredict()
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

