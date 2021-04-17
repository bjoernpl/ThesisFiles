# Tacotron 2 finetuning with ESPnet
We finetune the [pretrained model](kan-bayashi/vctk_tts_train_gst_tacotron2_raw_phn_tacotron_g2p_en_no_space_train.loss.best) from the [ESPnet model zoo](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv) by following
the ESPnet instructions carefully.

1. Prepare the Kaldi-style data directory according to [this example](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory). We provide a [notebook](https://github.com/bjoernpl/ThesisFiles/blob/main/DataPreparation/KaldiDataDirectory.ipynb) for this. 

2. Then follow the instructions on [TTS training](https://github.com/espnet/espnet/blob/a265764b9934d5b295edfaa8c4ff8f9b60c3292b/egs2/TEMPLATE/tts1/README.md) and [TTS finetuning](https://github.com/espnet/espnet/blob/a265764b9934d5b295edfaa8c4ff8f9b60c3292b/egs2/jvs/tts1/README.md). The training configuration is
based on [this example](https://github.com/espnet/espnet/blob/a265764b9934d5b295edfaa8c4ff8f9b60c3292b/egs2/vctk/tts1/conf/tuning/train_gst_tacotron2.yaml). Batch sizes must be adapted to fit your system.