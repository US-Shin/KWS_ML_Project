"""Matplotlib 그래프에서 한글 글꼴을 설정합니다."""

import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_korean_font():
    """사용 중인 운영체제에서 찾을 수 있는 한글 글꼴을 적용합니다."""
    candidates = [
        "Malgun Gothic",      # Windows 맑은 고딕
        "NanumGothic",        # 나눔고딕
        "Noto Sans CJK KR",   # Noto 한글 글꼴
        "AppleGothic",        # macOS 애플고딕
    ]

    for font_name in candidates:
        try:
            font_path = font_manager.findfont(
                font_name, fallback_to_default=False
            )
            font_manager.fontManager.addfont(font_path)
            resolved_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = resolved_name
            plt.rcParams["axes.unicode_minus"] = False
            return resolved_name
        except (ValueError, OSError):
            continue

    # 한글 글꼴을 찾지 못해도 그래프 생성은 계속합니다.
    plt.rcParams["axes.unicode_minus"] = False
    print("한글 글꼴을 찾지 못했습니다. Windows에서는 맑은 고딕 설치 여부를 확인하세요.")
    return None
