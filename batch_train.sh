#!/bin/bash

gpu=$1
data=$2

# Use Python to read the number of folds from config.json
end=$(python3 -c "import json; f=open('config.json'); print(json.load(f)['data_loader']['args']['num_folds']); f.close()")
end=$((end-1))

start=0

for i in $(eval echo {$start..$end})
do
   python3 train_Kfold_CV.py --fold_id=$i --device $gpu --np_data_dir $data
done
