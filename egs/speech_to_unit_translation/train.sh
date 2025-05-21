src=en
tgt=es
task=voxpopuli-${src}${tgt}-prefix
subtag=vp_ep_NormUnits_textless
arch=s2ut_transformer_textless

# if you want to use the prefix training, please use the follow arch
# arch=s2ut_transformer_textless_prefix

tag="your_tag_name"

ROOT = "root_path"
DATA_ROOT=${ROOT}/egs/speech_to_unit_translation/data/data_${src}2${tgt}_${subtag}/

config_dir=./config/
save_dir=checkpoints/${task}/${tag}
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

cp ${BASH_SOURCE[0]} ${save_dir}/train_ft.sh
cp -r ${DATA_ROOT}/${config_dir} ${save_dir}/config

n_cluster=1000
keep_last_epochs=10
bleu_valid=1

device=0,1,2,3,4,5,6,7
gpu_num=`echo "${device}" | awk '{split($0,arr,",");print length(arr)}'`
update_freq=1       
max_tokens=20000
max_update=100000   

learning_rate=0.0005   # pretrain: esen 0.0005 other 0.0003
dropout=0.3            # pretrain: esen 0.1 other 0.3

finetune=1

cmd="fairseq-train $DATA_ROOT
  --distributed-world-size ${gpu_num}
  --config-yaml ${config_dir}/config.yaml --multitask-config-yaml ${config_dir}/config_multitask.yaml
  --task speech_to_speech_bleu
  --target-is-code --target-code-size ${n_cluster} 
  --vocoder code_hifigan
  --criterion speech_to_unit --label-smoothing 0.2
  --arch ${arch} --share-decoder-input-output-embed
  --dropout ${dropout} --attention-dropout 0.2 --relu-dropout 0.2
  --train-subset train --valid-subset valid
  --save-dir ${save_dir}
  --lr ${learning_rate} --lr-scheduler inverse_sqrt
  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 10.0
  --max-update ${max_update} --max-tokens ${max_tokens} --max-target-positions 3000 --update-freq ${update_freq}
  --seed 1 --fp16 --num-workers 8
  --no-progress-bar
  --tensorboard-logdir ${save_dir}
  --skip-invalid-size-inputs-valid-test
  "
  
if [[ $bleu_valid -eq 1 ]]; then
    cmd="$cmd
    --eval-bleu
    --best-checkpoint-metric bleu
    --keep-best-checkpoints 10
    --maximize-best-checkpoint-metric"
fi
if [[ -n $keep_last_epochs ]]; then
    cmd="${cmd}
    --keep-last-epochs $keep_last_epochs "
fi

if [[ -n $finetune ]]; then
    cmd="${cmd}
    --finetune-from-model ${save_dir}/checkpoint_best.pretrain.pt "
fi

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" >> $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
