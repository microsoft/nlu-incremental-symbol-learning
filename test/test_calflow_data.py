# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import sys 
import os 
import pdb
import traceback

from allennlp.data.token_indexers.token_indexer import TokenIndexer

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

#from task_oriented_dialogue_as_dataflow_synthesis.src.dataflow.core.dialogue import Dialogue
from dataflow.core.lispress import parse_lispress, program_to_lispress, lispress_to_program, render_compact, render_pretty
from dataflow.core.dialogue import Dialogue
from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from miso.data.dataset_readers.calflow_parsing.calflow_sequence import CalFlowSequence
from miso.data.dataset_readers.calflow import CalFlowDatasetReader
from miso.data.tokenizers import MisoTokenizer

def assert_dict(produced, expected):
    for key in expected:
        assert(produced[key] == expected[key])

@pytest.fixture
def load_test_lispress():
    return """( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) ( PersonName.apply "Darby" ) ) ) ) ) ) )"""

@pytest.fixture
def load_long_lispress():
    return """(Yield (UpdateCommitEventWrapper (UpdatePreflightEventWrapper (Event.id (singleton (QueryEventResponse.results (FindEventWrapperWithDefaults (Event.attendees_? (AttendeeListHasRecipientConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "Matthew")))))))) (Event.start_? (?= (DateAtTimeWithDefaults (NextDOW (Thursday)) (DateTime.time (Event.end (singleton (QueryEventResponse.results (FindEventWrapperWithDefaults (Event.attendees_? (AttendeeListHasRecipientConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "Jeremy")))))))))))))))"""

@pytest.fixture
def load_let_lispress():
    return """(let (x0 (DateAtTimeWithDefaults (NextDOW (Monday)) (NumberAM 8L))) (Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (& (& (Event.start_? (?= x0)) (Event.end_? (?= (TimeAfterDateTime x0 (NumberAM 10L))))) (Event.showAs_? (?= (ShowAsStatus.Busy))))))))"""

@pytest.fixture
def load_path_lispress():
    return """(Yield (Execute (ReviseConstraint (refer (^(Dynamic) roleConstraint (Path.apply "output"))) (^(Event) ConstraintTypeIntension) (Event.start_? (DateTime.time_? (?= (ConvertTimeToPM (Execute (refer (& (^(Time) roleConstraint (Path.apply "start")) (extensionConstraint (^(Time) EmptyStructConstraint))))))))))))"""

@pytest.fixture
def load_do_singleton_lispress():
    return """(do (singleton (QueryEventResponse.results (FindEventWrapperWithDefaults (& (& (Event.subject_? (?~= "lunchdate")) (Event.start_? (DateTime.date_? (?= (NextDOW (Thursday)))))) (Event.attendees_? (AttendeeListHasRecipientConstraint (RecipientWithNameLike (^(Recipient) EmptyStructConstraint) (PersonName.apply "Lisa")))))))) (Yield (UpdateCommitEventWrapper (UpdatePreflightEventWrapper (Event.id (Execute (refer (extensionConstraint (^(Event) EmptyStructConstraint))))) (Event.start_? (DateTime.date_? (?= (ClosestDayOfWeek (DateTime.date (Event.start (Execute (refer (extensionConstraint (^(Event) EmptyStructConstraint)))))) (Friday)))))))))"""
    # return """(do (singleton (QueryEventResponse.results (FindEventWrapperWithDefaults (& (& (Event.subject? (?~= "lunchdate")) (Event.start? (DateTime.date? (?= (NextDOW (Thursday)))))) (Event.attendees? (AttendeeListHasRecipientConstraint (RecipientWithNameLike ((^(Recipient) EmptyStructConstraint)) (PersonName.apply "Lisa")))))))) (Yield (UpdateCommitEventWrapper (UpdatePreflightEventWrapper (Event.id (Execute (refer (extensionConstraint ((^(Event) EmptyStructConstraint)))))) (Event.start? (DateTime.date? (?= (ClosestDayOfWeek (DateTime.date (Event.start (Execute (refer (extensionConstraint ((^(Event) EmptyStructConstraint))))))) (Friday)))))))))"""

@pytest.fixture
def load_underlying_lispress():
    return """( Yield ( Execute ( NewClobber ( refer ( ^ ( Dynamic ) ActionIntensionConstraint ) ) ( ^ ( ( Constraint DateTime ) ) roleConstraint ( Path.apply "time" ) ) ( intension ( DateTime.date? ( ?= ( Tomorrow ) ) ) ) ) ) )"""

@pytest.fixture
def load_variable_order_lispress():
    return """( let ( x0 ( PersonName.apply "Elli Parker" ) ) ( do ( Yield ( Execute ( ChooseCreateEventFromConstraint ( ^ ( Event ) EmptyStructConstraint ) ( refer ( ^ ( Dynamic ) ActionIntensionConstraint ) ) ) ) ) ( Yield ( > ( size ( QueryEventResponse.results ( FindEventWrapperWithDefaults ( EventOnDate ( Tomorrow ) ( Event.attendees? ( & ( AttendeeListHasRecipientConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) x0 ) ) ( AttendeeListHasPeople ( FindTeamOf ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) x0 ) ) ) ) ) ) ) ) ) ) ) ) 0L ) ) ) )"""

@pytest.fixture
def load_reentrant_expression_lispress():
    return """( let ( x0 ( NextDOW ( Friday ) ) ) ( Yield ( CreateCommitEventWrapper ( CreatePreflightEventWrapper ( EventAllDayOnDate ( EventAllDayOnDate ( Event.subject? ( ?= "spending" ) ) x0 ) ( nextDayOfWeek x0 ( Sunday ) ) ) ) ) ) )"""

@pytest.fixture
def load_inf_loss_lispress():
    return """( let ( x0 ( DateAtTimeWithDefaults ( Execute ( refer ( extensionConstraint ( ^ ( Date ) EmptyStructConstraint ) ) ) ) ( Noon ) ) ) ( do "Shiro's sushi" ( Yield ( CreateCommitEventWrapper ( CreatePreflightEventWrapper ( & ( & ( & ( & ( & ( Event.subject_? ( ?= "lunch date" ) ) ( Event.start_? ( ?= x0 ) ) ) ( Event.end_? ( ?= ( TimeAfterDateTime x0 ( NumberPM 2L ) ) ) ) ) ( Event.location_? ( ?= ( LocationKeyphrase.apply "Shiro's sushi" ) ) ) ) ( Event.showAs_? ( ?= ( ShowAsStatus.OutOfOffice ) ) ) ) ( Event.attendees_? ( AttendeeListHasRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) ( PersonName.apply "Kate" ) ) ) ) ) ) ) ) ) ) ) ) )"""

@pytest.fixture
def load_all_valid_tgt_str():
    data_path = os.path.join(path, "data", "smcalflow.full.data", "valid.tgt") 
    with open(data_path) as f1:
       lines = f1.readlines() 
    return lines 

@pytest.fixture
def load_all_train_tgt_str():
    data_path = os.path.join(path, "data", "smcalflow.full.data", "train.tgt") 
    with open(data_path) as f1:
       lines = f1.readlines() 
    return lines 

# def test_tgt_str_to_list_short(load_test_lispress_short):
#     calflow_graph = CalFlowGraph(src_str="", tgt_str = load_test_lispress_short)
#     assert(calflow_graph.node_name_list == ['Yield', 'output', 'Execute', 'intension', 'ConfirmAndReturnAction'])
#     assert(calflow_graph.node_idx_list == [0, 1, 2, 3, 4])
#     assert(calflow_graph.edge_head_list == [0, 0, 1, 2, 3])
#     assert(calflow_graph.edge_type_list == ["fxn_arg-0", "arg-0", "fxn_arg-0", "arg-0", "fxn_arg-0"])

def calflow_roundtrip(test_str):
    true_ls_prog, __ = lispress_to_program(parse_lispress(test_str),0)
    print(true_ls_prog)
    true_lispress = program_to_lispress(true_ls_prog)

    calflow_graph = CalFlowGraph(src_str="", tgt_str = test_str)
    program = CalFlowGraph.prediction_to_program(calflow_graph.node_name_list, 
                                                calflow_graph.node_idx_list, 
                                                calflow_graph.edge_head_list, 
                                                calflow_graph.edge_type_list)

    pred_lispress = program_to_lispress(program)
    true_ls_prog, __ = lispress_to_program(parse_lispress(test_str),0)
    true_lispress = program_to_lispress(true_ls_prog)
    true_lispress_str = render_pretty(true_lispress)
    pred_lispress_str = render_pretty(pred_lispress)
    assert(pred_lispress_str == true_lispress_str)
    # except (AssertionError,KeyError,IndexError, AttributeError, TypeError) as error:
   

def test_calflow_roundtrip_base(load_test_lispress):
    calflow_roundtrip(load_test_lispress)
    
def test_calflow_roundtrip_long(load_long_lispress):
    calflow_roundtrip(load_long_lispress)

def test_calflow_roundtrip_path(load_path_lispress):
    calflow_roundtrip(load_path_lispress)

def test_calflow_roundtrip_singleton(load_do_singleton_lispress):
    calflow_roundtrip(load_do_singleton_lispress)

def test_calflow_roundtrip_nested_underlying(load_underlying_lispress):
    calflow_roundtrip(load_underlying_lispress)

def test_calflow_roundtrip_let(load_let_lispress):
    calflow_roundtrip(load_let_lispress) 

def test_calflow_roundtrip_variable_order(load_variable_order_lispress):
    calflow_roundtrip(load_variable_order_lispress)

def test_calflow_roundtrip_expression_order(load_reentrant_expression_lispress):
    calflow_roundtrip(load_reentrant_expression_lispress)

def test_calflow_rountrip_inf_issue(load_inf_loss_lispress):
    calflow_roundtrip(load_inf_loss_lispress)

@pytest.mark.skip(reason="too large")
def test_calflow_roundtrip_valid(load_all_valid_tgt_str):
    all_lines = load_all_valid_tgt_str
    skipped = 0
    mistakes = 0
    for i, line in enumerate(all_lines):
        try:
            line=line.strip()
            lispress_from_line = parse_lispress(line)
            clean_true_lispress_str = render_compact(lispress_from_line) 
            calflow_graph = CalFlowGraph(src_str="", tgt_str = clean_true_lispress_str)
            program = CalFlowGraph.prediction_to_program(calflow_graph.node_name_list, 
                                                        calflow_graph.node_idx_list,
                                                        calflow_graph.edge_head_list, 
                                                        calflow_graph.edge_type_list) 
            pred_lispress = program_to_lispress(program)
            pred_lispress = program_to_lispress(lispress_to_program(pred_lispress, 0)[0])
            # run through again to get sugaring
            true_ls_prog, __ = lispress_to_program(lispress_from_line,0)
            true_lispress = program_to_lispress(true_ls_prog)
            true_lispress_str = render_pretty(true_lispress)
            pred_lispress_str = render_pretty(pred_lispress)

            assert(pred_lispress_str == true_lispress_str)

        except (AssertionError, IndexError, KeyError) as e:
            progress = i/len(all_lines)
            print(progress)
            print(pred_lispress_str)
            print(true_lispress_str)
            pdb.set_trace() 
            mistakes += 1

@pytest.mark.skip(reason="too large")
def test_calflow_roundtrip_train(load_all_train_tgt_str):
    all_lines = load_all_train_tgt_str
    skipped = 0
    mistakes = 0
    for i, line in enumerate(all_lines):
        try:
            line=line.strip()
            lispress_from_line = parse_lispress(line)
            clean_true_lispress_str = render_compact(lispress_from_line) 
            calflow_graph = CalFlowGraph(src_str="", tgt_str = clean_true_lispress_str)
            program = CalFlowGraph.prediction_to_program(calflow_graph.node_name_list, 
                                                        calflow_graph.node_idx_list,
                                                        calflow_graph.edge_head_list, 
                                                        calflow_graph.edge_type_list) 
            pred_lispress = program_to_lispress(program)
            pred_lispress = program_to_lispress(lispress_to_program(pred_lispress, 0)[0])
            # run through again to get sugaring
            true_ls_prog, __ = lispress_to_program(lispress_from_line,0)
            true_lispress = program_to_lispress(true_ls_prog)
            true_lispress_str = render_pretty(true_lispress)
            pred_lispress_str = render_pretty(pred_lispress)

            assert(pred_lispress_str == true_lispress_str)

        except (AssertionError, IndexError, KeyError) as e:
            progress = i/len(all_lines)
            print(progress)
            print(pred_lispress_str)
            print(true_lispress_str)
            pdb.set_trace() 
            mistakes += 1


@pytest.fixture
def load_seq_strings_basic():
    #src_str = """__User Darby __StartOfProgram ( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ( ^ ( Recipient ) EmptyStructConstraint ) ) ( PersonName.apply "Darby" ) ) ) ) ) ) ) __User Dirty Dan __StartOfProgram"""
    src_str = """__User Darby __StartOfProgram"""
    tgt_str = """( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ( ^ ( Recipient ) EmptyStructConstraint ) ) ( PersonName.apply "Darby" ) ) ) ) ) ) )"""
    return (src_str, tgt_str)

def test_calflow_get_list_data(load_seq_strings_basic):
    src_str, tgt_str = load_seq_strings_basic
    g = CalFlowGraph(src_str = src_str,
                     tgt_str = tgt_str)
    data = g.get_list_data(bos="@start@",
                           eos="@end@")


@pytest.mark.skip(reason="deprecated")
def test_calflow_sequence_basic(load_seq_strings_basic):
    src_str, tgt_str = load_seq_strings_basic
    seq_obj = CalFlowSequence(src_str, tgt_str, use_program=False)
    output = seq_obj.get_list_data(bos="@start@", eos="@end@")

    assert(output['src_copy_inds'] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1])
    assert(output['tgt_tokens'][output['src_copy_inds'].index(2)] == "Darby")


@pytest.fixture
def load_indexers():
    source_params = {"source_token_characters": {
                        "type": "characters",
                        "min_padding_length": 5,
                        "namespace": "source_token_characters"
                    },
                    "source_tokens": {
                        "type": "single_id",
                        "namespace": "source_tokens"
                        }
                    }
    target_params = {"target_token_characters": {
                        "type": "characters",
                        "min_padding_length": 5,
                        "namespace": "target_token_characters"
                    },
                    "target_tokens": {
                        "type": "single_id",
                        "namespace": "target_tokens"
                        }
                    }

    source_token_indexers = {k: TokenIndexer(**params) for k, params in source_params.items()}
    target_token_indexers = {k: TokenIndexer(**params) for k, params in target_params.items()}

    return source_token_indexers, target_token_indexers


@pytest.mark.skip(reason="deprecated")
def test_calflow_dataset_reader(load_indexers):
    source_token_indexers, target_token_indexers = load_indexers
    data_path = os.path.join(path, "data", "smcalflow.full.data", "tiny.dataflow_dialogues.jsonl")
    generation_token_indexers = target_token_indexers
    tokenizer = MisoTokenizer()
    evaluation = False 
    line_limit = None
    lazy = False

    dataset_reader = CalFlowDatasetReader(source_token_indexers,
                                          target_token_indexers,
                                          generation_token_indexers,
                                          tokenizer,
                                          evaluation,
                                          line_limit,
                                          lazy)

    data = dataset_reader._read(data_path)

    assert(data)
