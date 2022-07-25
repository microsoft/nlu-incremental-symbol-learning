# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import *
import re
import logging

import torch
from allennlp.common.from_params import Params, T
from allennlp.training.optimizers import Optimizer

logger = logging.getLogger('optim')


@Optimizer.register('transformer')
class TransformerOptimizer:
    """
    Wrapper for AllenNLP optimizer.
    This is used to fine-tune the pretrained transformer with some layers fixed and different learning rate.
    When some layers are fixed, the wrapper will set the `require_grad` flag as  False, which could save
    training time and optimize memory usage.
    Plz contact Guanghui Qin for bugs.
    Params:
        base: base optimizer.
        embeddings_lr: learning rate for embedding layer. Set as 0.0 to fix it.
        encoder_lr: learning rate for encoder layer. Set as 0.0 to fix it.
        pooler_lr: learning rate for pooler layer. Set as 0.0 to fix it.
        layer_fix: the number of encoder layers that should be fixed.

    Example json config:

    1. No-op. Do nothing (why do you use me?)
    optimizer: {
        type: "transformer",
        base: {
            type: "adam",
            lr: 0.001
        }
    }

    2. Fix everything in the transformer.
    optimizer: {
        type: "transformer",
        base: {
            type: "adam",
            lr: 0.001
        },
        embeddings_lr: 0.0,
        encoder_lr: 0.0,
        pooler_lr: 0.0
    }

    Or equivalently (suppose we have 24 layers)

    optimizer: {
        type: "transformer",
        base: {
            type: "adam",
            lr: 0.001
        },
        embeddings_lr: 0.0,
        layer_fix: 24,
        pooler_lr: 0.0
    }

    3. Fix embeddings and the lower 12 encoder layers, set a small learning rate
       for the other parts of the transformer

    optimizer: {
        type: "transformer",
        base: {
            type: "adam",
            lr: 0.001
        },
        embeddings_lr: 0.0,
        layer_fix: 12,
        encoder_lr: 1e-5,
        pooler_lr: 1e-5
    }
    """
    @classmethod
    def from_params(
            cls: Type[T],
            params: Params,
            model_parameters: List[Tuple[str, torch.nn.Parameter]],
            **_
    ):
        param_groups = list()

        def remove_param(keyword_):
            nonlocal model_parameters
            logger.info(f'Fix param with name matching {keyword_}.')
            for name, param in model_parameters:
                if keyword_ in name:
                    logger.debug(f'Fix param {name}.')
                    param.requires_grad_(False)
            model_parameters = list(filter(lambda x: keyword_ not in x[0], model_parameters))

        for i_layer in range(params.pop('layer_fix')):
            remove_param('transformer_model.encoder.layer.{}.'.format(i_layer))

        for specific_lr, keyword in (
            (params.pop('embeddings_lr', None), 'transformer_model.embeddings'),
            (params.pop('encoder_lr', None), 'transformer_model.encoder.layer'),
            (params.pop('pooler_lr', None), 'transformer_model.pooler'),
        ):
            if specific_lr is not None:
                if specific_lr > 0.:
                    pattern = '.*' + keyword.replace('.', r'\.') + '.*'
                    if len([name for name, _ in model_parameters if re.match(pattern, name)]) > 0:
                        param_groups.append([[pattern], {'lr': specific_lr}])
                    else:
                        logger.warning(f'{pattern} is set to use lr {specific_lr} but no param matches.')
                else:
                    remove_param(keyword)

        param_groups.extend(params.pop('parameter_groups', list()))

        return Optimizer.by_name(params.get('base').pop('type'))(
            model_parameters=model_parameters, parameter_groups=param_groups,
            **params.pop('base').as_flat_dict()
        )

