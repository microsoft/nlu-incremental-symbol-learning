# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


cfg=amlt_configs/transformer/FindManager_12_seed/5000_200.yaml

for base in 5000 10000 20000 50000 100000 max
do
	for num in 50 200 500
	do
		new_cfg=amlt_configs/transformer/Tomorrow_12_seed/${base}_${num}.yaml
		cp ${cfg} ${new_cfg}
		sed -i "s/5000_200/${base}_${num}/g" ${new_cfg}
		sed -i "s/FindManager/Tomorrow/g" ${new_cfg}
	done
done
