# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json 
import sys 

with open(sys.argv[1]) as f1, open(sys.argv[2], "w") as f2:
    for line in f1.readlines():
        l = json.loads(line)
        l2 = {"datum_id": l}
        l2_str = json.dumps(l2) 
        f2.write(l2_str + "\n")


