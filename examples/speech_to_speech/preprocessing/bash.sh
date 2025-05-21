AUDIO_DIR=/workspace/data/zhangyuhao/voxpopuli/audio/transcribed_data/en/$1/wav
AUIDO_EXT=wav
GEN_SUBSET=train_enfr_$1
DATA_DIR=/workspace/data/zhangyuhao/vp-zyh/en-norm
python prep_sn_data.py \
	  --audio-dir ${AUDIO_DIR} --ext ${AUIDO_EXT} \
	  --data-name ${GEN_SUBSET} --output-dir ${DATA_DIR} \
	  --for-inference
