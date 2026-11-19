"""원본 데이터와 캐시를 제외하고 제출 ZIP을 패키징합니다."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "기말-202235271-신의서.zip"
INCLUDE = [
    "README.md", "submission.json", "requirements.txt", ".gitignore",
    "app", "src", "models", "results", "deployment", "screenshots",
    "tools", "output/pdf/포트폴리오.pdf",
]
EXCLUDED_DIRS = {".git", ".gradio", "__pycache__", ".venv", "data"}


def main():
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=7) as archive:
        for item in INCLUDE:
            path = ROOT / item
            if path.is_file():
                archive.write(path, path.relative_to(ROOT).as_posix())
                continue
            if not path.is_dir():
                continue
            for file in sorted(path.rglob("*")):
                if not file.is_file():
                    continue
                if any(part in EXCLUDED_DIRS for part in file.parts):
                    continue
                if file.suffix == ".pyc" or file.name.startswith("page-"):
                    continue
                archive.write(file, file.relative_to(ROOT).as_posix())

    with ZipFile(OUTPUT) as archive:
        members = archive.namelist()
        assert not any(name.startswith("data/") for name in members)
        assert any(name.endswith("포트폴리오.pdf") for name in members)
        assert any(name == "screenshots/README.md" for name in members)
    print(f"ZIP: {OUTPUT}")
    print(f"BYTES: {OUTPUT.stat().st_size}")
    print(f"FILES: {len(members)}")
    print(f"SERVICE_SCREENSHOTS: {sum(name.startswith('screenshots/') and name.endswith('.png') for name in members)}")


if __name__ == "__main__":
    main()
