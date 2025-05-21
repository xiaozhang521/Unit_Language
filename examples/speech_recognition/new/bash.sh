GEN_SUBSET=train_enfr_2014
RESULTS_PATH=/workspace/data/zhangyuhao/vp-zyh/en-norm/$GEN_SUBSET
mkdir -p ${RESULTS_PATH}
DATA_DIR=/workspace/data/zhangyuhao/vp-zyh/en-norm
eval="python3 infer.py \
    --config-dir /workspace/data/zhangyuhao/fairseq-S2ST/examples/hubert/config/decode/ \
    --config-name infer_viterbi \
    task.data=${DATA_DIR} \
    task.normalize=false \
    common_eval.results_path=${RESULTS_PATH}/log \
    common_eval.path=/workspace/data/zhangyuhao/vp-zyh/sn/en_10h/checkpoint_best.pt \
    dataset.gen_subset=${GEN_SUBSET} \
    '+task.labels=["unit"]' \
    +decoding.results_path=${RESULTS_PATH} \
    common_eval.post_process=none \
    +dataset.batch_size=1 \
    common_eval.quiet=True"
echo $eval
CUDA_VISIBLE_DEVICES=0 eval $eval
