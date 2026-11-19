"""
Gradio Demo App for Keyword Spotting Service Prototype
Loads the trained proposed model and performs real-time inference from microphone or uploaded audio.
"""

import gradio as gr
import torch
import numpy as np
import librosa
import pickle
from src.proposed_model import ProposedResCNN
from src.data_preparation import preprocess_audio, CLASSES, CLASS_TO_IDX

# Load model and class info
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ProposedResCNN(num_classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load("models/proposed_rescnn.pth", map_location=device))
model.eval()

def predict_keyword(audio):
    if audio is None:
        return "음성을 입력해주세요."
    
    sr, audio_array = audio
    # Gradio returns (sample_rate, numpy array)
    
    try:
        # Preprocess
        features = preprocess_audio(audio_array.astype(np.float32), sr)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, time)
        
        with torch.no_grad():
            logits = model(features_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
        
        predicted_class = CLASSES[pred_idx]
        
        result = f"**예측 키워드**: {predicted_class}\n**신뢰도**: {confidence:.2%}"
        
        # Top-3
        top3 = torch.topk(probs, 3)
        top3_str = "\n".join([f"{CLASSES[i]}: {p:.2%}" for i, p in zip(top3.indices, top3.values)])
        result += f"\n\n**Top-3**:\n{top3_str}"
        
        return result
    except Exception as e:
        return f"오류 발생: {str(e)}"

# Build Gradio Interface
demo = gr.Interface(
    fn=predict_keyword,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="음성 입력"),
    outputs=gr.Markdown(label="인식 결과"),
    title="🎤 Keyword Spotting Demo - 스마트홈 명령어 인식",
    description="""
    학습된 모델로 음성 명령어를 인식합니다.
    
    **지원 키워드**: yes, no, on, off, stop, go, up, down, left, right, unknown, silence
    
    마이크를 사용하거나 오디오 파일을 업로드하세요.
    """,
    examples=None,
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
