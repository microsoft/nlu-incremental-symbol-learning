# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import torch
from collections import defaultdict

class GroupDROLoss(torch.nn.CrossEntropyLoss):
    def __init__(self):
        super(GroupDROLoss, self).__init__(reduction='none')

    def forward(self, pred_logits, true_classes): 
        # compute loss per instance
        loss_per_instance = super().forward(pred_logits, true_classes)
        # get per-group loss 
        group_idxs = defaultdict(list)
        for bidx in range(true_classes.shape[0]):
            tc = true_classes[bidx].item() 
            group_idxs[tc].append(bidx)

        max_loss = 0.0
        max_group = -1
        for group in group_idxs.keys():
            loss_idxs = group_idxs[group]
            avg_loss = torch.mean(loss_per_instance[loss_idxs])
            if avg_loss > max_loss:
                max_group = group
                max_loss = avg_loss


        #print(f"Max group: {max_group} with loss {max_loss.item()}")
        return max_loss 


