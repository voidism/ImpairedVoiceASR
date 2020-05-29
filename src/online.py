import torch
from src.data import create_transform

# collect_audio_batch(batch, audio_transform, mode)
# audio_transform_dv, feat_dim_dv = create_transform(audio.copy(), dv=True)
# tokenizer = load_text_encoder(**text)
# tr_set, dv_set, tr_loader_bs, dv_loader_bs, mode, data_msg = create_dataset(
#        tokenizer, ascending, **corpus)

class Datadealer():
    def __init__(self, audio_setting):
        self.audio_transform, feat_dim = create_transform(audio_setting.copy(), dv=True)

    def __call__(self, wav_file_name):
        feat = self.audio_transform(wav_file_name)
        audio_len = len(feat)
        return feat.unsqueeze(0), torch.tensor([audio_len])
