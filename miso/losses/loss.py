# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.nn import MSELoss, L1Loss, BCELoss
from allennlp.common import Registrable


class LossFunctionDict(dict):
    def __init__(self,*arg,**kwargs):
        super(LossFunctionDict, self).__init__(*arg, **kwargs)
        self['MSELoss'] = MSELoss()
        self['L1Loss'] = L1Loss()
        self['MSECrossEntropyLoss'] = MSECrossEntropyLoss()

class Loss(torch.nn.Module, Registrable): 
    def __init__(self):
        super(Loss, self).__init__()
        pass

    def forward(self, output, target):
        pass 

@Loss.register("mse_cross_entropy") 
class MSECrossEntropyLoss(Loss): 
    def __init__(self):
        super(MSECrossEntropyLoss, self).__init__()
        self.mse_criterion = MSELoss()
        self.xent_criterion = BCELoss()

    def forward(self, output, target):
        mse_value = self.mse_criterion(output, target)
        
        thresholded_output = torch.gt(output, 0).float()
        thresholded_target = torch.gt(target, 0).float()

        xent_value = self.xent_criterion(thresholded_output, thresholded_target)
        if xent_value + mse_value != 0:
            harmonic_mean = 2*(xent_value * mse_value)/(xent_value + mse_value)
        else:
            # 0 by default, all 0's
            harmonic_mean = xent_value + mse_value
        return harmonic_mean

@Loss.register("group_dro") 
class GroupDroLoss(Loss):
    def __init__(self):
        super(MSECrossEntropyLoss, self).__init__()
        self.xent_criterion = BCELoss(reduction='none')

    def forward(self, output, target):
        pass 