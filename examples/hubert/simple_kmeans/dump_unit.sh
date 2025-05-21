feat_dir=/workspace/data/zhangyuhao/voxpopuli/audio/transcribed_data/fr-en/unit_tmp/fr/origin/feature
nshard=10
km_path=/workspace/data/zhangyuhao/vp-zyh/mhubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
lab_dir=/workspace/data/zhangyuhao/vp-zyh/fr-unit
split=fr_audio_w_root
for rank in $(seq 0 $((nshard - 1)));
do
python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
#echo $rank
done
