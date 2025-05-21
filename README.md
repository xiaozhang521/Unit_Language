# Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation

We provide the implementation proposed in our paper for texeless S2S(Speech-to-Speech) translation task.  By proposing the unit language, we overcome two main challenges, the cross-model(CM) modeling and the cross-lingual(CL) modeling, in speech-to-speech process.

This code is fork from https://github.com/facebookresearch/fairseq/tree/v0.12.3.


## Project Structure
```
egs
|---speech_to_unit_translation
|    |---train.sh
|    |---decode.sh
|    |---evaluate.sh
|    |---conf
|    |---|---config.yaml
```


## Data Processing

### Unit Generation

For the data processing scripts, please follow the guide at https://github.com/facebookresearch/fairseq/blob/v0.12.3/examples/speech_to_speech/docs/textless_s2st_real_data.md.

### Unit Language Generation

To Generate the Unit language, we adopt the n-gram or BPE method.

## Getting Started

To train or evaluate the model, please follow the guide at https://github.com/facebookresearch/fairseq/blob/v0.12.3/examples/speech_to_speech/docs/direct_s2st_discrete_units.md.

### Multitask data

For each multitask `$TASK_NAME`, prepare `${DATA_ROOT}/${TASK_NAME}/${SPLIT}.tsv` files for each split following the format below:

```
id  tgt_text
sample_id_0 token1 token2 token3 ...
sample_id_1 token1 token2 token3 ...
...
```

For each multitask `$TASK_NAME`, prepare `${DATA_ROOT}/${TASK_NAME}/dict.txt`, a dictionary in fairseq format with all tokens for the targets for `$TASK_NAME`.


Copy the `conf/config.yaml` file to `${DATA_ROOT}/${TASK_NAME}/config/`, and create `config_multitask.yaml` including three encoder multitasks (source_unit, source_text (for unit language or text), target_text (for unit language or text)). 

```
source_unit:
  decoder_type: transformer
  dict: ${DATA_ROOT}/source_unit/dict.txt
  data: ${DATA_ROOT}/source_unit
  encoder_layer: 6
  loss_weight: 0.5
source_text:
  decoder_type: transformer
  dict: ${DATA_ROOT}/source_text/dict.txt
  data: ${DATA_ROOT}/source_text
  encoder_layer: 8
  loss_weight: 0.5
target_text:
  decoder_type: transformer
  dict: ${DATA_ROOT}/target_text/dict.txt
  data: ${DATA_ROOT}/target_text
  encoder_layer: 12
  loss_weight: 0.5
```

### Model Training

Run `speech_to_unit_translation/train.sh` to train the model. Before training, please modify the necessary paths and tags in the `train.sh`. 

```bash
bash train.sh
```

### Model Inference

Adjust all the required paths and tags in the `speech_to_unit_translation/decode.sh` to match the `train.sh` file, and run `speech_to_unit_translation/decode.sh` to inference the trained model.

```bash
bash decode.sh
```

### Unit-to-waveform conversion with unit vocoder and evaluate

Run `evaluate.sh` to generate wavs according to units and make evaluations. Before the runing, please download the `VOCODER` and `wav2vec2` model to the target path.

```
bash evaluate.sh
```

