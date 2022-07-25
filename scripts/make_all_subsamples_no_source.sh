# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash 


PlaceHasFeature_triggers="takeout,casual,waiter"
./scripts/make_subsamples_no_source.sh PlaceHasFeature ~/resources/data/smcalflow.agent.data ~/resources/data/smcalflow_samples_no_source ${PlaceHasFeature_triggers}

FindManager_triggers="boss,manager,supervisor"
./scripts/make_subsamples_no_source.sh FindManager ~/resources/data/smcalflow.agent.data ~/resources/data/smcalflow_samples_no_source ${FindManager_triggers}

Tomorrow_triggers="tomorrow"
./scripts/make_subsamples_no_source.sh Tomorrow ~/resources/data/smcalflow.agent.data ~/resources/data/smcalflow_samples_no_source ${Tomorrow_triggers}

FenceAttendee_triggers="meet,mom"
./scripts/make_subsamples_no_source.sh FenceAttendee ~/resources/data/smcalflow.agent.data ~/resources/data/smcalflow_samples_no_source ${FenceAttendee_triggers}

DoNotConfirm_triggers="cancel,n't,no"
./scripts/make_subsamples_no_source.sh DoNotConfirm ~/resources/data/smcalflow.agent.data ~/resources/data/smcalflow_samples_no_source ${DoNotConfirm_triggers}


