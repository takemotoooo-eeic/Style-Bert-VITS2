from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)

from style_bert_vits2.tts_model import TTSModelHolder
import torch
from config import get_path_config
from scipy.io.wavfile import write
import time

def tts_fn(
        text: str,
    ):
        start = time.time()
        
        model_holder.get_model("0618_ryotsu", "model_assets/0618_ryotsu/0618_ryotsu_e72_s1000.safetensors")
        assert model_holder.current_model is not None

        speaker_id = model_holder.current_model.spk2id["0618_ryotsu"]

        load = time.time()

        try:
            sr, audio = model_holder.current_model.infer(
                text=text,
                language=Languages.JP,
                reference_audio_path=None,
                sdp_ratio=DEFAULT_SDP_RATIO,
                noise=DEFAULT_NOISE,
                noise_w=DEFAULT_NOISEW,
                length=DEFAULT_LENGTH,
                line_split=DEFAULT_LINE_SPLIT,
                split_interval=DEFAULT_SPLIT_INTERVAL,
                assist_text="",
                assist_text_weight=DEFAULT_ASSIST_TEXT_WEIGHT,
                use_assist_text=False,
                style=DEFAULT_STYLE,
                style_weight=DEFAULT_STYLE_WEIGHT,
                given_tone=None,
                speaker_id=speaker_id,
                pitch_scale=1,
                intonation_scale=1,
            )
            end = time.time()
            print(load - start)
            print(end - load)
        except Exception as e:
             raise(e)
        return sr, audio

if __name__ == "__main__":

    file_path = "test.wav"
    sentence = "こち亀記念館にようこそ！"

    now =time.time()
    
    path_config = get_path_config()
    assets_root = path_config.assets_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_holder = TTSModelHolder(assets_root, device)
    print("初期化")
    print(time.time() - now)

    response = tts_fn(sentence)
    print(response)
    write(file_path, response[0], response[1])
