# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env python
# encoding: utf-8


class CandidateMappings:

    def __init__(self):
        self._candidate_mappings = []

    def __str__(self):
        return "\t" + "\n\t".join(
            "%d: [%s]" %(key, str(value))
            for key, value in enumerate(self._candidate_mappings)
        )

    def __getitem__(self, key):
        return self._candidate_mappings[key]

    def append(self, value):
        self._candidate_mappings.append(value)

    def sort(self):
        for key, value in enumerate(self._candidate_mappings):
            self._candidate_mappings[key] = sorted(value)

    def items(self):
        return enumerate(self._candidate_mappings)
