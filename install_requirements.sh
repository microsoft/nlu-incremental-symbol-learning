# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

pip install -r requirements.txt 

git clone git@github.com:microsoft/task_oriented_dialogue_as_dataflow_synthesis.git
cd task_oriented_dialogue_as_dataflow_synthesis
python setup.py build
python setup.py install
cd ..
rm -rf task_oriented_dialogue_as_dataflow_synthesis

