#! /usr/bin/bash
st="r" results_dir="results"
visual_encoder="stgcn" language_encoder="clip-vit-b-32" tm="chat"
mode="train"
dataset=$1

if [ "$dataset" = "ntu60" ]; then
    ls=96 ils=8 lr=4.9372938499672305e-05 batch_size=32 dis_step=4
    th=0 t=0
    num_classes=60 nc=10 nepc=1700
    ss=5
    available_splits=("split2" "split3" "split4")
elif [ "$dataset" = "ntu120" ]; then
    ls=304 ils=12 lr=2.133571484619993e-05 batch_size=24 dis_step=16
    th=0 t=0
    num_classes=120 nc=10 nepc=1700
    ss=10
    available_splits=("split2" "split3" "split4")
else
    echo "Dataset not supported"
    exit 1
fi

run_experiment() {
    ss=$1
    dataset_local=$2
    tdir="resources/sk_feats/${visual_encoder}_${dataset_local}_${ss}_r/"
    edir="resources/sk_feats/${visual_encoder}_${dataset_local}_val_${ss}_r/"
    wdir_1="results/${visual_encoder}_${dataset_local}_${ss}_r/"
    wdir_2="results/${visual_encoder}_${dataset_local}_val_${ss}_r/"

    echo "-----------------------------------"
    echo "=========="
    echo "Stage 1"
    echo "..."
    r1=$(
        python train.py \
            --num_classes $num_classes --ss "$ss" --st $st --ve $visual_encoder --le $language_encoder --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc \
            --latent_size $ls --i_latent_size $ils --lr $lr --phase train --mode $mode --dataset_path "$tdir" --wdir "$wdir_1" \
            --dis_step $dis_step --batch_size $batch_size --dataset $dataset_local
    )
    za=${r1:0-35:5} c=${r1:0-18:1}
    echo "Best ZSL Acc: $za on cycle $c"

    echo "=========="
    echo "Stage 2"
    echo "..."
    r2=$(
        python train.py \
            --num_classes $num_classes --ss "$ss" --st $st --ve $visual_encoder --le $language_encoder --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc \
            --latent_size $ls --i_latent_size $ils --lr $lr --phase val --mode $mode --dataset_path "$edir" --wdir "$wdir_2" \
            --dis_step $dis_step --batch_size $batch_size --dataset $dataset_local
    )

    echo "=========="
    echo "Stage 3"
    echo "..."
    r3=$(
        python gating_train.py \
            --num_classes $num_classes --ss "$ss" --st $st --ve $visual_encoder --le $language_encoder --tm $tm --phase val --dataset_path "$edir" \
            --wdir "$wdir_2" --th $th --t $t --dataset $dataset_local
    )
    echo "thresh: ${r3:0-23:4}, temp: ${r3:0-1}"

    echo "=========="
    echo "Stage 4"
    echo "..."
    r4=$(
        python gating_eval.py \
            --num_classes $num_classes --ss "$ss" --st $st --phase train --dataset_path "$tdir" --wdir "$wdir_1" --ve $visual_encoder --le $language_encoder --tm $tm \
            --thresh "${r3:0-23:4}" --temp "${r3:0-1}" --dataset $dataset_local
    )
    sa=${r4:15:5} ua=${r4:39:5} hm=${r4:0-6:5}
    echo "S_Acc: ${sa}, U_Acc: ${ua}, H_Mean: ${hm}"
    echo "Finish"
}

for dataset_split in "${available_splits[@]}"; do
    run_experiment $ss $dataset"_"$dataset_split
done
