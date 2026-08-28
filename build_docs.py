"""portfolio.html → docs/index.html (GitHub Pages 용 독립 실행 HTML).

portfolio.html 은 <html>/<head>/<body> 껍데기 없이 본문만 담고 있다.
GitHub Pages 로 서빙하려면 문서 골격과 최소 리셋을 씌워야 한다.
'이력서 문구'와 '면접 대비'처럼 작업용으로 쓴 섹션은 공개본에서 제외한다.
"""
import pathlib
import re

SRC = pathlib.Path("portfolio.html")
OUT = pathlib.Path("docs/index.html")

# 공개본에서 뺄 섹션 (eyebrow 라벨로 식별)
DROP_SECTIONS = ["포트폴리오", "면접 대비"]

HEAD_EXTRA = """  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="투구 단위 상태로 예측하는 KBO 실시간 승률과 그 근거">
  <style>
    *,*::before,*::after{box-sizing:border-box}
    html{-webkit-text-size-adjust:100%}
    body,h1,h2,h3,h4,p,figure,blockquote,dl,dd{margin:0}
    img,picture,svg{max-width:100%;display:block}
    table{border-collapse:collapse}
  </style>"""


def drop_sections(html: str) -> str:
    for label in DROP_SECTIONS:
        while True:
            m = re.search(r'<section>(?:(?!</section>).)*?<span class="eyebrow">'
                          + re.escape(label) + r'</span>.*?</section>', html, re.S)
            if not m:
                break
            html = html[:m.start()] + html[m.end():]
    return html


def main() -> None:
    src = drop_sections(SRC.read_text(encoding="utf-8"))
    head, body = [], []
    for ln in src.split("\n"):
        st = ln.strip()
        (head if st.startswith("<title>") or st.startswith("<link rel=") else body).append(ln)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        '<!doctype html>\n<html lang="ko">\n<head>\n'
        + HEAD_EXTRA + "\n"
        + "\n".join("  " + l.strip() for l in head if l.strip())
        + "\n</head>\n<body>\n"
        + "\n".join(body).strip()
        + "\n</body>\n</html>\n",
        encoding="utf-8")
    print(f"docs/index.html · {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
