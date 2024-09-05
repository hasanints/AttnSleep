gpu=$1
data=$2

# Menggunakan Python untuk membaca nilai num_folds
end=$(python3 -c "import json; print(json.load(open('config.json'))['data_loader']['args']['num_folds'])")
end=$((end-1))

for i in $(eval echo {$start..$end})
do
    python3 train_Kfold_CV.py --fold_id=$i --device $gpu --np_data_dir $data
done