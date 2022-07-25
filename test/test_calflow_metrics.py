# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import sys 
import os 

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 
from dataflow.core.lispress import parse_lispress, render_compact
from miso.metrics.fxn_metrics import SingleFunctionMetric


def test_function_metric_fine_correct():
    test_str = """( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) ( PersonName.apply "Darby" ) ) ) ) ) ) )"""
    test_str = render_compact(parse_lispress(test_str))
    fxn_name = "RecipientWithNameLike"
    pred_str = test_str
    metric = SingleFunctionMetric(fxn_name)
    metric(test_str, pred_str)
    coarse, fine, __, __, __ = metric.get_metric()
    assert(fine == 1.0)

def test_function_metric_incorrect():
    test_str = """( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) ( PersonName.apply "Darby" ) ) ) ) ) ) )"""
    test_str = render_compact(parse_lispress(test_str))
    fxn_name = "RecipientWithNameLike"
    pred_str = """( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( EventOnDateWithTimeRange ( ^ ( Recipient ) EmptyStructConstraint ) ( PersonName.apply "Darby" ) ) ) ) ) ) )"""
    pred_str = render_compact(parse_lispress(pred_str))
    metric = SingleFunctionMetric(fxn_name)
    metric(test_str, pred_str)
    coarse, fine, __, __, __ = metric.get_metric()
    assert(fine == 0.0)
    assert(coarse == 0.0)

def test_function_metric_coarse_correct():
    test_str = """( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) ( PersonName.apply "Darby" ) ) ) ) ) ) )"""
    test_str = render_compact(parse_lispress(test_str))
    fxn_name = "RecipientWithNameLike"
    pred_str = """( Yield ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) ) )"""
    pred_str = render_compact(parse_lispress(pred_str))
    metric = SingleFunctionMetric(fxn_name)
    metric(test_str, pred_str)
    coarse, fine, __, __, __ = metric.get_metric()
    assert(coarse == 1.0)
    assert(fine == 0.0)
