"""Hugging Face Spaces inference app for the trained keyword spotter."""

from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch

from src.data_preparation import CLASSES, SILENCE_RMS_THRESHOLD, preprocess_audio
from src.proposed_model import ProposedResCNN

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = ProposedResCNN(num_classes=len(CLASSES)).to(DEVICE)
MODEL.load_state_dict(
    torch.load(ROOT / "models" / "proposed_rescnn.pth", map_location=DEVICE)
)
MODEL.eval()

MIN_CONFIDENCE = 0.65
MIN_MARGIN = 0.15


def predict_keyword(audio):
    """입력 음성을 전처리하고 확신도가 충분한 키워드만 반환합니다."""
    if audio is None:
        return "음성을 입력해주세요."

    sample_rate, audio_array = audio
    try:
        audio_array = np.asarray(audio_array)
        if np.issubdtype(audio_array.dtype, np.integer):
            audio_float = audio_array.astype(np.float32) / np.iinfo(audio_array.dtype).max
        else:
            audio_float = audio_array.astype(np.float32)
            if audio_float.size and np.max(np.abs(audio_float)) > 1.5:
                audio_float /= 32768.0
        if audio_float.ndim > 1:
            audio_float = np.mean(audio_float, axis=1)

        rms = float(np.sqrt(np.mean(audio_float**2) + 1e-12))
        if rms < SILENCE_RMS_THRESHOLD:
            return "음성이 감지되지 않았습니다. 마이크 거리와 입력 음량을 확인해주세요."

        features = preprocess_audio(audio_float, sample_rate, crop_active=True)
        tensor = torch.as_tensor(features).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.softmax(MODEL(tensor), dim=1)[0]

        top = torch.topk(probabilities, 3)
        predicted = CLASSES[int(top.indices[0])]
        confidence = float(top.values[0])
        margin = float(top.values[0] - top.values[1])
        top_text = "\n".join(
            f"{CLASSES[int(index)]}: {float(probability):.2%}"
            for index, probability in zip(top.indices, top.values)
        )

        if predicted == "silence":
            return f"음성이 아닌 것으로 판단했습니다.\n\nTop-3:\n{top_text}"
        if confidence < MIN_CONFIDENCE or margin < MIN_MARGIN:
            return f"확신하기 어려워 키워드를 결정하지 않았습니다. 다시 또렷하게 말해주세요.\n\nTop-3:\n{top_text}"
        return f"**예측 키워드**: {predicted}\n\n**신뢰도**: {confidence:.2%}\n\n**Top-3**:\n{top_text}"
    except Exception as error:
        return f"음성을 처리하지 못했습니다: {error}"


demo = gr.Interface(
    fn=predict_keyword,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="음성 입력"),
    outputs=gr.Markdown(label="인식 결과"),
    title="🎙️ 스마트홈 키워드 인식",
    description=(
        "영어 명령어 yes/no/on/off/stop/go/up/down/left/right 중 한 단어를 "
        "짧고 또렷하게 말하거나 WAV 음성을 업로드하세요. 낮은 음량이나 불확실한 "
        "입력은 예측을 보류합니다."
    ),
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
