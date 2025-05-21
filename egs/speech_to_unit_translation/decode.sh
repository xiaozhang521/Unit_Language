ROOT = "root_path"
workspace=${ROOT}/egs/speech_to_unit_translation/
device=0

src=es    
tgt=en

task=voxpopuli-${src}${tgt}-prefix

tag="your_tag_name"

pretrained_model_dir=${ROOT}/egs/speech_to_unit_translation/pretrained/

VOCODER_TAG=mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj
VOCODER_CKPT=${pretrained_model_dir}/vocoders/${VOCODER_TAG}/g_00500000
VOCODER_CFG=${pretrained_model_dir}/vocoders/${VOCODER_TAG}/config.json

subtag=vp_ep_NoNorm_textless

DATA_ROOT=${ROOT}/egs/speech_to_unit_translation/data/data_${src}2${tgt}_${subtag}/
MODEL_DIR=${pretrained_model_dir}/asr_models/models--facebook--wav2vec2-large-960h-lv60-self/snapshots/54074b1c16f4de6a5ad59affb4caa8f2ea03a119

GEN_SUBSET=test 
checkpoint_tag=checkpoint_best.pt

config_dir=./config/
save_dir=checkpoints/${task}/${tag}
script_dir=${workspace}/script/
wav_dir=${save_dir}/wav/prediction/
cp ${BASH_SOURCE[0]} ${save_dir}/decode.sh

export CUDA_VISIBLE_DEVICES=$device 

fairseq-generate $DATA_ROOT \
  --config-yaml ${config_dir}/config.yaml --multitask-config-yaml ${config_dir}/config_multitask.yaml \
  --task speech_to_speech --target-is-code --target-code-size 1000 --vocoder code_hifigan \
  --path $save_dir/${checkpoint_tag}  --gen-subset $GEN_SUBSET \
  --max-tokens 20000 \
  --beam 10 --max-len-a 1 \
  --results-path ${save_dir} \
  --skip-invalid-size-inputs-valid-test \
  --max-target-positions 3000 \