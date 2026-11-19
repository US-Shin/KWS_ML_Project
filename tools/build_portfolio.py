"""README의 주요 내용과 현재 평가 그림으로 포트폴리오 PDF를 만듭니다."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "포트폴리오.pdf"
FONT = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("Malgun", str(FONT / "malgun.ttf")))
pdfmetrics.registerFont(TTFont("Malgun-Bold", str(FONT / "malgunbd.ttf")))

NAVY = colors.HexColor("#133B65")
BLUE = colors.HexColor("#2878B5")
PALE = colors.HexColor("#EAF2F8")
GRAY = colors.HexColor("#536273")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Cover", fontName="Malgun-Bold", fontSize=25, leading=35, textColor=NAVY, alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle(name="Sub", fontName="Malgun", fontSize=12, leading=19, textColor=GRAY, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="H1K", fontName="Malgun-Bold", fontSize=17, leading=23, textColor=NAVY, spaceAfter=8))
styles.add(ParagraphStyle(name="H2K", fontName="Malgun-Bold", fontSize=11, leading=15, textColor=BLUE, spaceBefore=6, spaceAfter=3))
styles.add(ParagraphStyle(name="BodyK", fontName="Malgun", fontSize=9, leading=14, textColor=colors.HexColor("#243447"), spaceAfter=4))
styles.add(ParagraphStyle(name="SmallK", fontName="Malgun", fontSize=7.7, leading=11, textColor=GRAY, spaceAfter=3))
styles.add(ParagraphStyle(name="CellK", fontName="Malgun", fontSize=8, leading=11, textColor=colors.HexColor("#243447")))
styles.add(ParagraphStyle(name="CellBoldK", fontName="Malgun-Bold", fontSize=8, leading=11, textColor=NAVY))


def P(text, style="BodyK"):
    return Paragraph(text, styles[style])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6E0EA"))
    canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
    canvas.setFont("Malgun", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(18 * mm, 10 * mm, "스마트 홈 음성 명령 키워드 스포팅 | 202235271 신의서")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def add_image(story, filename, width, height):
    image = Image(str(ROOT / "results" / filename), width=width * mm, height=height * mm)
    image.hAlign = "CENTER"
    story.append(image)


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=17*mm, bottomMargin=21*mm, title="키워드 스포팅 프로젝트 포트폴리오", author="신의서")
    story = [Spacer(1, 36*mm), P("스마트 홈 음성 명령을 위한<br/>키워드 스포팅", "Cover"), P("기계학습프로그래밍 기말 프로젝트", "Sub"), Spacer(1, 20*mm)]
    cover = Table([[P("학번", "CellBoldK"), P("202235271", "CellK")], [P("이름", "CellBoldK"), P("신의서", "CellK")], [P("프로젝트", "CellBoldK"), P("영어 음성 명령 12개 클래스 분류", "CellK")]], colWidths=[40*mm, 90*mm], rowHeights=[12*mm]*3)
    cover.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), PALE), ("BOX", (0,0), (-1,-1), .5, colors.HexColor("#CCD9E5")), ("INNERGRID", (0,0), (-1,-1), .35, colors.HexColor("#D7E1EA")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 10)]))
    story += [cover, Spacer(1, 18*mm), P("테스트 정확도 93.52%, Macro-F1 93.48%", "Sub"), Spacer(1, 30*mm), P("GitHub: https://github.com/US-Shin/ML_Projects_KWS", "SmallK"), P("임시 공개 데모: https://6d91600116fe8bdd9e.gradio.live", "SmallK"), P("앱 실행 중에만 이용 가능하며 최대 1주일 후 만료될 수 있습니다.", "SmallK"), PageBreak()]

    story += [P("1. 문제와 접근 방법", "H1K"), P("짧은 영어 음성 명령을 분류해 스마트홈 제어에 활용하는 키워드 스포팅 시스템입니다. 마이크 또는 파일 입력에서 키워드를 예측하고, 음량이 낮거나 확신도가 부족하면 결과를 보류합니다."), P("데이터와 전처리", "H2K"), P("Google Speech Commands v0.02 원본을 TensorFlow 공식 다운로드 주소에서 받았습니다. 타깃 키워드 10개와 unknown, silence를 합쳐 12개 클래스입니다. 타깃 단어는 클래스당 최대 300개를 사용하고, silence는 배경 소음으로 구성합니다. 라이선스는 CC BY 4.0입니다."), P("모노 변환과 16 kHz 리샘플링 후 40 Mel 필터 로그 Mel 스펙트로그램을 생성합니다. 입력은 1초로 맞춥니다. 학습/검증/테스트는 계층화 70/15/15 분할이며 random_state=42입니다."), P("모델 비교", "H2K")]
    rows = [[P("항목", "CellBoldK"), P("Baseline CNN", "CellBoldK"), P("Proposed ResCNN", "CellBoldK")], [P("구조", "CellK"), P("3개 합성곱 층 + 분류기", "CellK"), P("잔차 블록 3개 + 분류기", "CellK")], [P("학습", "CellK"), P("처음부터 학습", "CellK"), P("처음부터 학습", "CellK")], [P("정확도", "CellK"), P("77.96%", "CellK"), P("93.52%", "CellK")], [P("Macro-F1", "CellK"), P("77.76%", "CellK"), P("93.48%", "CellK")]]
    table = Table(rows, colWidths=[34*mm, 62*mm, 62*mm], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), PALE), ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#CCD9E5")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 7), ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6)]))
    story += [table, Spacer(1, 6*mm), P("평가 결과", "H1K")]
    add_image(story, "model_comparison.png", 165, 66)
    story += [P("두 모델은 같은 테스트셋에서 비교했습니다. 제안 ResCNN은 정확도와 클래스별 성능 균형을 반영하는 Macro-F1에서 베이스라인보다 높았습니다. 실제 마이크 환경에서는 별도 검증이 필요합니다.", "SmallK"), PageBreak()]

    story += [P("2. 결과 그림 읽기", "H1K"), P("학습 곡선", "H2K")]
    add_image(story, "proposed_training_curve.png", 168, 61)
    story += [P("왼쪽은 학습/검증 손실, 가운데는 검증 정확도, 오른쪽은 검증 Macro-F1입니다. 점선과 표식은 검증 Macro-F1이 가장 높았던 에폭으로, 해당 가중치를 체크포인트로 저장해 평가와 추론에 사용했습니다.", "SmallK"), P("혼동행렬", "H2K")]
    add_image(story, "proposed_rescnn_confusion_matrix.png", 156, 76)
    story += [P("세로축은 실제 라벨, 가로축은 예측 라벨입니다. 대각선은 정답이고 대각선 밖은 오분류입니다. 클래스별 샘플 수를 고려해 개수와 비율을 함께 확인합니다.", "SmallK"), P("서비스 및 제출 상태", "H2K"), P("Gradio 앱은 `app/demo.py`, Hugging Face Spaces 배포 파일은 `deployment/hf_space/`에 있습니다. 임시 공개 데모는 위 주소에서 확인할 수 있지만 앱 프로세스가 종료되거나 링크가 만료되면 접근할 수 없습니다. 영구 Space는 계정 인증 후 배포해야 합니다. 브라우저 화면 캡처 3장은 아직 추가되지 않았으며 `screenshots/README.md`에 필요한 파일 목록을 적었습니다.", "BodyK"), P("학습 데이터 원본(약 2.43 GB)과 전처리 캐시는 제출 ZIP에서 제외했습니다. 코드 실행 시 데이터 준비 스크립트가 원본을 내려받습니다.", "SmallK")]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
