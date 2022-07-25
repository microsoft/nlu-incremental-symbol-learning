# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Iterator, Optional
import argparse
import sys
import json
from overrides import overrides 
from collections import defaultdict 
import pdb 
import pickle as pkl 
import spacy 
from spacy.tokenizer import Tokenizer

from allennlp.commands.predict import _get_predictor, Predict
from allennlp.commands import ArgumentParserWithDefaults
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance
from allennlp.commands.predict import _PredictManager
from allennlp.common.util import import_submodules

from miso.predictors.decomp_parsing_predictor import sanitize, DecompSyntaxParsingPredictor
from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph
from miso.data.dataset_readers.decomp_parsing.decomp_with_syntax import DecompGraphWithSyntax

#from decomp import UDSVisualization, serve_parser

def parse_api_sentence(input_line, args, predictor):
    #semantics_only = args.semantics_only
    #drop_syntax = args.drop_syntax
    
    manager = _ReturningPredictManager(
                                predictor = predictor,
                                input_file = input_line,
                                output_file = None,
                                batch_size = 1,
                                print_to_console = False,
                                has_dataset_reader = True,
                                beam_size = 2,
                                line_limit = 1,
                                oracle = False,
                                json_output_file = None)

    manager._dataset_reader.api_time = True

    if isinstance(predictor, DecompSyntaxParsingPredictor):
        sem_graph, syn_graph, __ = manager.run()[1][0]
        return DecompGraphWithSyntax.arbor_to_uds(sem_graph, syn_graph, "test-graph", input_line) 

    return DecompGraph.arbor_to_uds(manager.run()[1][0], "test-graph", input_line) 

def get_input_lines(input_file):
    SPACY_MODEL = "en_core_web_sm"
    nlp = spacy.load(SPACY_MODEL)
    tokenizer = Tokenizer(nlp.vocab)
    with open(input_file) as f1:
        return [str(tokenizer(x.strip())) for x in f1.readlines()]

def _predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)

    if args.run_api:
        serve_parser(lambda x: parse_api_sentence(x, args, predictor), with_syntax=with_syntax) 

    elif args.run_arbitrary:
        input_lines = get_input_lines(args.input_file)
        
        output_graphs = []
        for line in input_lines:
            output_graph = parse_api_sentence(line, args, predictor) 
            output_graphs.append(output_graph)
        with open(args.output_file, 'wb') as f1:
            pkl.dump(output_graphs, f1) 

    else:
        if args.silent and not args.output_file:
            print("--silent specified without --output-file.")
            print("Exiting early because no output will be created.")
            sys.exit(0)

        manager = _ReturningPredictManager(
                                    predictor = predictor,
                                    input_file = args.input_file,
                                    output_file = args.output_file,
                                    batch_size = args.batch_size,
                                    print_to_console = False,
                                    has_dataset_reader = True,
                                    beam_size = args.beam_size,
                                    line_limit = args.line_limit,
                                    oracle = args.oracle, 
                                    json_output_file = None)

        manager.run()

class _ReturningPredictManager(_PredictManager):
    """
    Extends the _PredictManager class to be able to return data
    which is required for spr scoring, to avoid unneccessary IO
    """
    def __init__(self,
                 predictor: Predictor,
                 input_file: str,
                 output_file: Optional[str],
                 batch_size: int,
                 print_to_console: bool,
                 has_dataset_reader: bool,
                 beam_size: int,
                 line_limit: int = None,
                 oracle: bool = False,
                 json_output_file: str = None) -> None:
        super(_ReturningPredictManager, self).__init__(predictor,
                                                       input_file,
                                                       None,
                                                       batch_size,
                                                       False,
                                                       has_dataset_reader)
        self.beam_size = beam_size
        self.line_limit = line_limit 
        self.oracle = oracle 
        self._json_output_file = json_output_file

    @overrides
    def _predict_instances(self, batch):
        # if not oracle, back off to _PredictManager default prediction 
        if not self.oracle:
            return super()._predict_instances(batch)
        else:
            results = self._predictor.predict_batch_instance(batch, self.oracle)
            return [results]

    def run(self):
        has_reader = self._dataset_reader is not None
        instances, results = [], []
        if has_reader:
            self._dataset_reader.line_limit = self.line_limit
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    instances.append(model_input_instance)
                    results.append(result)

        # if oracle, unify all dicts
        if self.oracle:
            # results: List[List[Dict]]
            final_dict = defaultdict(lambda: defaultdict(dict))
            for res_dict in results:
                #res_dict = res_dict[0]
                for prop_key in res_dict.keys():
                    try:
                        final_dict[prop_key]['true_val_with_node_ids'].update(res_dict[prop_key]['true_val_with_node_ids'])
                        final_dict[prop_key]['pred_val_with_node_ids'].update(res_dict[prop_key]['pred_val_with_node_ids'])
                    except KeyError:
                        # edge attributes
                        final_dict[prop_key]['true_val_with_edge_ids'].update(res_dict[prop_key]['true_val_with_edge_ids'])
                        final_dict[prop_key]['pred_val_with_edge_ids'].update(res_dict[prop_key]['pred_val_with_edge_ids'])

            if self._json_output_file is not None:
                with open(self._json_output_file, "w") as f1:
                    json.dump(sanitize(final_dict), f1)

        return instances, results


class _CalFlowReturningPredictManager(_ReturningPredictManager):
    """
    Extends the _PredictManager class to be able to return data
    which is required for spr scoring, to avoid unneccessary IO
    """
    def __init__(self,
                 predictor: Predictor,
                 input_file: str,
                 output_file: Optional[str],
                 batch_size: int,
                 print_to_console: bool,
                 has_dataset_reader: bool,
                 beam_size: int,
                 line_limit: int = None,
                 oracle: bool = False,
                 top_k_beam_search: bool = False,
                 top_k: int = 1, 
                 precomputed: bool = False) -> None:
        super(_CalFlowReturningPredictManager, self).__init__(predictor=predictor,
                                                       input_file=input_file,
                                                       output_file=output_file,
                                                       batch_size=batch_size,
                                                       print_to_console=print_to_console,
                                                       has_dataset_reader=has_dataset_reader,
                                                       beam_size=beam_size,
                                                       line_limit=line_limit)
        self.precomputed = precomputed
        self.oracle = oracle
        self.top_k_beam_search = top_k_beam_search
        self.top_k = top_k

    @overrides
    def _predict_instances(self, batch):
        # if not oracle or top k, back off to _PredictManager default prediction 
        if not self.oracle and not self.top_k_beam_search:
            return super()._predict_instances(batch)
        elif not self.oracle and self.top_k_beam_search:
            results = self._predictor.predict_batch_instance(batch, self.oracle, self.top_k_beam_search, self.top_k)
            return [results]
        elif self.oracle and not self.top_k_beam_search:
            results = self._predictor.predict_batch_instance(batch, self.oracle) 
            return [results]
        else:
            raise AssertionError() 

    def run(self):
        has_reader = self._dataset_reader is not None
        instances, results = [], []
        if has_reader:
            self._dataset_reader.line_limit = self.line_limit
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                if not self.precomputed:
                    for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                        if not self.top_k_beam_search:
                            instances.append(model_input_instance)
                            results.append(result)
                        else:
                            for i in range(len(result)):
                                instances.append(model_input_instance)
                                results.append(result[i])
                else:
                    for input_instance in batch:
                        instances.append(input_instance)
        return instances, results


class Predict(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('input_file', type=str, help='path to or url of the input file')

        subparser.add_argument('--output-file', type=str, help='path to output file')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('--use-dataset-reader',
                               action='store_true',
                               help='Whether to use the dataset reader of the original model to load Instances. '
                                    'The validation dataset reader will be used if it exists, otherwise it will '
                                    'fall back to the train dataset reader. This behavior can be overridden '
                                    'with the --dataset-reader-choice flag.')

        subparser.add_argument('--dataset-reader-choice',
                               type=str,
                               choices=['train', 'validation'],
                               default='validation',
                               help='Indicates which model dataset reader to use if the --use-dataset-reader '
                                    'flag is set.')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.add_argument("--run-api",
                                action="store_true",
                                help="set to true to run an online API" )

        subparser.add_argument("--run-arbitrary",
                                action="store_true",
                                help="set to true to run an arbitrary sentences sored in input file" )

        subparser.add_argument("--beam-size",
                                type=int,
                                default=1)
        subparser.add_argument("--line-limit", 
                                type=int,
                                default=None)
        subparser.add_argument("--oracle", action = "store_true", help="run with forced decode")

        subparser.set_defaults(func=_predict)

        return subparser


    
if __name__ == "__main__":
    parser = ArgumentParserWithDefaults(description="Run AllenNLP")
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "predict": Predict(),
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
