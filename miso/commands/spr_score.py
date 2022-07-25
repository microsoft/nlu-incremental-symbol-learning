# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Iterator, Optional
import argparse
import sys
import json

import torch
import numpy as np
import random

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

from miso.predictors import  DecompPredictor
from miso.metrics.s_metric import utils
from miso.commands.predict import _ReturningPredictManager 

torch.manual_seed(12) 
np.random.seed(12) 
random.seed(12) 


def _predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    manager = _ReturningPredictManager(predictor,
                              args.input_file,
                              args.output_file,
                              args.batch_size,
                              not args.silent,
                              args.use_dataset_reader,
                              args.beam_size,
                              oracle = args.oracle,
                              json_output_file = args.json_output_file,
                              line_limit = args.line_limit)
    manager.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Use a trained model to make predictions.')

    parser.add_argument('--archive-file', required=True, type=str, help='the archived model to make predictions with')
    parser.add_argument('--input-file', required=True, type=str, help='path to input file')

    parser.add_argument('--output-file', type=str, help='path to output file')
    parser.add_argument('--json-output-file', type=str, required = True, help='path to dump json of spr attributes ')
    parser.add_argument('--weights-file', type=str, help='a path that overrides which weights file to use')

    parser.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

    parser.add_argument('--silent', action='store_true', help='do not print output to stdout')

    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('--line-limit', type=int, default=None, help='how many lines to run (debugging)')

    parser.add_argument('--use-dataset-reader',
                           action='store_true',
                           help='Whether to use the dataset reader of the original model to load Instances')

    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a JSON structure used to override the experiment configuration')

    parser.add_argument('--predictor',
                           type=str,
                           help='optionally specify a specific predictor to use')

    parser.add_argument('--beam-size',
                        type=int,
                        default=1,
                        help="Beam size for seq2seq decoding")
    parser.add_argument("--oracle", 
                        action="store_true",
                        help="set to true to use oracle gold graphs and only predict SPR properties"),

    args = parser.parse_args()

    if args.cuda_device >= 0:
        device = torch.device('cuda:{}'.format(args.cuda_device))
    else:
        device = torch.device('cpu')
    args.cuda_device = device

    _predict(args)

