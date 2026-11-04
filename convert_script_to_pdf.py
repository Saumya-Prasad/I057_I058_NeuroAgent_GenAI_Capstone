import os
import re
import subprocess
import markdown
import pymupdf

BASE_DIR = r"c:\Users\Saumya_Shambhavi\Downloads\GEN AI + CAPSTONE"
MD_PATH = os.path.join(BASE_DIR, "NeuroAgent_Presentation_Speaker_Script.md")
HTML_PATH = os.path.join(BASE_DIR, "NeuroAgent_Presentation_Speaker_Script.html")
PDF_PATH = os.path.join(BASE_DIR, "NeuroAgent_Presentation_Speaker_Script.pdf")

with open(MD_PATH, "r", encoding="utf-8") as f:
    raw_md = f.read()

# Preprocess GitHub alerts into HTML styled callouts
def alert_sub(match):
    alert_type = match.group(1).upper()
    body = match.group(2).strip()
    colors = {
        "IMPORTANT": ("#7c3aed", "#f5f3ff", "🟣 IMPORTANT DIRECTIVE"),
        "TIP": ("#2563eb", "#eff6ff", "💡 PRESENTATION TIP"),
        "WARNING": ("#dc2626", "#fef2f2", "⚠️ CRITICAL WARNING"),
        "NOTE": ("#0891b2", "#ecfeff", "ℹ️ NOTE")
    }
    border, bg, title = colors.get(alert_type, ("#4b5563", "#f9fafb", "NOTICE"))
    body_html = "<br>".join(line.lstrip("> ").strip() for line in body.split("\n"))
    return f"""<div style="border-left: 4px solid {border}; background: {bg}; padding: 12px 18px; margin: 16px 0; border-radius: 6px;">
<div style="font-weight: 700; color: {border}; margin-bottom: 6px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
<div style="font-size: 13.5px; color: #1f2937; line-height: 1.5;">{body_html}</div>
</div>"""

md_processed = re.sub(r'>\s*\[!(IMPORTANT|TIP|WARNING|NOTE)\]\n((?:>.*\n?)+)', alert_sub, raw_md)

# Convert Markdown to HTML
html_content = markdown.markdown(
    md_processed,
    extensions=["tables", "fenced_code", "nl2br"]
)

# Custom styling for high-end academic presentation document
css = """
@page {
    size: letter portrait;
    margin: 18mm 18mm 20mm 18mm;
    @bottom-right {
        content: counter(page);
    }
}

* {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    line-height: 1.6;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 12px;
}

h2 {
    font-size: 18px;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 6px;
    margin-top: 28px;
    margin-bottom: 14px;
    page-break-after: avoid;
}

h3 {
    font-size: 15px;
    font-weight: 700;
    color: #0369a1;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
}

h4 {
    font-size: 13.5px;
    font-weight: 700;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 10px;
}

blockquote {
    border-left: 4px solid #2563eb;
    background-color: #f8fafc;
    margin: 10px 0 16px 0;
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: #0f172a;
    line-height: 1.65;
    font-style: normal;
}

blockquote p {
    margin-bottom: 8px;
}

blockquote p:last-child {
    margin-bottom: 0;
}

code {
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 11.5px;
    background-color: #f1f5f9;
    color: #0f766e;
    padding: 2px 5px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
}

pre {
    background-color: #0f172a;
    color: #f8fafc;
    padding: 12px 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 11.5px;
    line-height: 1.45;
}

pre code {
    background-color: transparent;
    color: inherit;
    padding: 0;
    border: none;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 12px;
}

th, td {
    padding: 8px 10px;
    border: 1px solid #cbd5e1;
    text-align: left;
}

th {
    background-color: #f1f5f9;
    font-weight: 700;
    color: #0f172a;
}

tr:nth-child(even) td {
    background-color: #f8fafc;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 10px;
    padding-left: 24px;
}

li {
    margin-bottom: 4px;
}

hr {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 22px 0;
}

.slide-block {
    page-break-inside: avoid;
    margin-bottom: 20px;
}

.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: #ffffff;
    padding: 20px 24px;
    border-radius: 8px;
    margin-bottom: 24px;
}

.header-banner h1 {
    color: #ffffff;
    border-bottom: none;
    margin: 0 0 6px 0;
}

.header-meta {
    font-size: 12.5px;
    color: #93c5fd;
    line-height: 1.5;
}
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeuroAgent: Capstone Presentation Speaker Script & Viva Guide</title>
<style>
{css}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(full_html)

print("Generated HTML successfully at:", HTML_PATH)

# Convert to PDF via headless Microsoft Edge
edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_exe):
    edge_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

print("Using browser binary:", edge_exe)

cmd = [
    edge_exe,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_PATH}",
    HTML_PATH
]

res = subprocess.run(cmd, capture_output=True, text=True)
print("Subprocess return code:", res.returncode)

if os.path.exists(PDF_PATH):
    doc = pymupdf.open(PDF_PATH)
    page_count = len(doc)
    size_mb = os.path.getsize(PDF_PATH) / (1024 * 1024)
    print(f"SUCCESS: Generated PDF '{PDF_PATH}' ({page_count} pages, {size_mb:.2f} MB)")
else:
    print("FAILED: PDF was not generated.")

