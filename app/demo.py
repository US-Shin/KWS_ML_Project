"""학습한 키워드 인식 모델을 마이크/파일 입력으로 실행하는 Gradio 앱."""

import sys
from pathlib import Path
import gradio as gr
import torch
import numpy as np
import librosa
import pickle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.proposed_model import ProposedResCNN
from src.data_preparation import preprocess_audio, CLASSES, SILENCE_RMS_THRESHOLD

# 모델과 클래스 정보를 불러오기.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ProposedResCNN(num_classes=len(CLASSES)).to(device)
model_path = ROOT / "models" / "proposed_rescnn.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 확률이 낮거나 모호한 예측 -> 키워드 지정x
MIN_CONFIDENCE = 0.65
MIN_MARGIN = 0.15

def predict_keyword(audio):
    if audio is None:
        return "음성을 입력해주세요."
    
    sr, audio_array = audio
    
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

        # 평균 음량이 낮으면 무음으로 판정.
        rms = float(np.sqrt(np.mean(audio_float ** 2) + 1e-12))
        if rms < SILENCE_RMS_THRESHOLD:
            return "음성이 감지되지 않았습니다. 마이크 거리와 입력 음량을 확인해주세요."

        # 에너지가 큰 1초를 사용
        features = preprocess_audio(audio_float, sr, crop_active=True)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, time)
        
        with torch.no_grad():
            logits = model(features_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
        
        predicted_class = CLASSES[pred_idx]
        top3 = torch.topk(probs, 3)
        top3_str = "\n".join([f"{CLASSES[i]}: {p:.2%}" for i, p in zip(top3.indices, top3.values)])

        if predicted_class == "silence":
            return f"음성이 아닌 것으로 판단했습니다.\n\nTop-3:\n{top3_str}"
        if confidence < MIN_CONFIDENCE or (top3.values[0] - top3.values[1]).item() < MIN_MARGIN:
            return f"확신하기 어려워 키워드를 결정하지 않았습니다. 다시 또렷하게 말해주세요.\n\nTop-3:\n{top3_str}"

        result = f"**예측 키워드**: {predicted_class}\n**신뢰도**: {confidence:.2%}"
        result += f"\n\n**Top-3**:\n{top3_str}"
        
        return result
    except Exception as e:
        return f"오류 발생: {str(e)}"


demo = gr.Interface(
    fn=predict_keyword,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="음성 입력"),
    outputs=gr.Markdown(label="인식 결과"),
    title="🎤 Keyword Spotting Demo - 스마트홈 명령어 인식",
    description="지원 단어는 영어 yes/no/on/off/stop/go/up/down/left/right 입니다. 한 단어를 짧고 또렷하게 말하세요. 한국어 문장이나 다른 단어는 인식하지 않습니다.",
    examples=None,
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
