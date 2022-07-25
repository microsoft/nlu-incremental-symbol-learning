# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import json 
import _jsonnet 

if __name__ == "__main__": 
    config_to_edit_path = sys.argv[1]
    jsonnet_config_path = sys.argv[2]

    with open(config_to_edit_path) as f1:
        config_to_edit = json.load(f1) 

    jsonnet_config = json.loads(_jsonnet.evaluate_file(jsonnet_config_path) )

    for key in ["random_seed", "numpy_seed", "pytorch_seed"]:
        config_to_edit[key] = jsonnet_config[key]

    with open(config_to_edit_path, "w") as f1:
        json.dump(config_to_edit, f1, indent=4) 

