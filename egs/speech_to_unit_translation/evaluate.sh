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

grep "^D\-" ${save_dir}/generate-${GEN_SUBSET}.txt | \
  sed 's/^D-//ig' | cut -f3 \
  > ${save_dir}/generate-${GEN_SUBSET}.unit


if [ ! -d "$wav_dir" ]; then
    mkdir -p "$wav_dir"
fi

rm -rf ${wav_dir}/*

python3 examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${save_dir}/generate-${GEN_SUBSET}.unit \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path ${wav_dir} --dur-prediction


LOGFILE=${save_dir}/asr_bleu_result.log
python3 ${script_dir}/eval_asr_bleu.py -i ${save_dir}/generate-${GEN_SUBSET}.txt -r ${DATA_ROOT}/test.tsv -d ${wav_dir} -m ${MODEL_DIR} -l ${tgt} > $LOGFILE

tail -n 10 $LOGFILE
