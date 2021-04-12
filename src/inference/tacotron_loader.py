import torch
from espnet_model_zoo.downloader import ModelDownloader
# Note that espnet2 must contain forked changes to allow direct acceptance of GSTs
from espnet2.tasks.tts import TTSTask

class TacotronLoader:
    @staticmethod
    def load():
        """
        Load Tacotron model for TTS inference. Also gives preprocessing function for text.
        """
        d = ModelDownloader('./models/')

        config_weights = d.download_and_unpack("kan-bayashi/vctk_tts_train_gst_tacotron2_raw_phn_tacotron_g2p_en_no_space_train.loss.best")
        tt2, train_args = TTSTask.build_model_from_file(
            config_weights["train_config"], config_weights["model_file"], "cuda"
        )
        tt2.use_gst = True
        _ = tt2.eval()

        preprocess = TTSTask.build_preprocess_fn(train_args, False)
        def preprocess_text(text):
            return torch.from_numpy(preprocess("<dummy>", {"text": text})["text"]).cuda()

        return tt2, preprocess_text