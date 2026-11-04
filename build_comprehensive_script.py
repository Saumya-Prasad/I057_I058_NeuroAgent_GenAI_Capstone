import os
import re
import subprocess
import markdown
import pymupdf

from script_generator.part1 import get_part1_text
from script_generator.part2 import get_part2_text
from script_generator.part3 import get_part3_text
from script_generator.part4 import get_part4_text
from script_generator.part5 import get_part5_text
from script_generator.part6 import get_part6_text

BASE_DIR = r"c:\Users\Saumya_Shambhavi\Downloads\GEN AI + CAPSTONE"
MD_PATH = os.path.join(BASE_DIR, "NeuroAgent_Master_Capstone_Script_Comprehensive.md")
HTML_PATH = os.path.join(BASE_DIR, "NeuroAgent_Master_Capstone_Script_Comprehensive.html")
PDF_PATH = os.path.join(BASE_DIR, "NeuroAgent_Master_Capstone_Script_Comprehensive.pdf")
ARTIFACT_DIR = r"C:\Users\Saumya_Shambhavi\.gemini\antigravity\brain\5ed7677f-f111-4333-a9f7-b21b42e20fc1"
ARTIFACT_PDF_PATH = os.path.join(ARTIFACT_DIR, "NeuroAgent_Master_Capstone_Script_Comprehensive.pdf")

print("Assembling comprehensive 40-slide markdown text...")
full_md = "\n\n".join([
    get_part1_text(),
    get_part2_text(),
    get_part3_text(),
    get_part4_text(),
    get_part5_text(),
    get_part6_text()
])

with open(MD_PATH, "w", encoding="utf-8") as f:
    f.write(full_md)

print(f"Saved Markdown file: {MD_PATH} ({len(full_md)} characters)")

# Preprocess GitHub-style alerts into HTML styled callouts
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

md_processed = re.sub(r'>\s*\[!(IMPORTANT|TIP|WARNING|NOTE)\]\n((?:>.*\n?)+)', alert_sub, full_md)

# Convert Markdown to HTML
html_content = markdown.markdown(
    md_processed,
    extensions=["tables", "fenced_code", "nl2br"]
)

# Custom print CSS
css = """
@page {
    size: letter portrait;
    margin: 16mm 16mm 18mm 16mm;
    @bottom-right {
        content: "Page " counter(page);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 10px;
        color: #64748b;
    }
    @bottom-left {
        content: "NeuroAgent: Master's Capstone Presentation Script & Viva Guide (2026)";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 10px;
        color: #64748b;
    }
}

* {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px;
    line-height: 1.55;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 12px;
}

h2 {
    font-size: 16px;
    font-weight: 800;
    color: #1e3a8a;
    background-color: #f1f5f9;
    border-left: 5px solid #1e3a8a;
    padding: 8px 12px;
    margin-top: 26px;
    margin-bottom: 14px;
    page-break-after: avoid;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

h3 {
    font-size: 14px;
    font-weight: 700;
    color: #0369a1;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
}

h4 {
    font-size: 12.5px;
    font-weight: 700;
    color: #334155;
    margin-top: 12px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 8px;
}

blockquote {
    border-left: 4px solid #2563eb;
    background-color: #f8fafc;
    margin: 8px 0 12px 0;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 12px;
    color: #0f172a;
    line-height: 1.6;
    font-style: normal;
}

blockquote p {
    margin-bottom: 6px;
}

blockquote p:last-child {
    margin-bottom: 0;
}

code {
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 11px;
    background-color: #f1f5f9;
    color: #0f766e;
    padding: 2px 4px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
}

pre {
    background-color: #0f172a;
    color: #f8fafc;
    padding: 10px 14px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 11px;
    line-height: 1.4;
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
    margin: 12px 0;
    font-size: 11.5px;
}

th, td {
    padding: 6px 8px;
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
    margin-bottom: 8px;
    padding-left: 20px;
}

li {
    margin-bottom: 3px;
}

hr {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 18px 0;
}

.slide-block {
    page-break-inside: avoid;
    margin-bottom: 16px;
}
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeuroAgent: 40-Slide Capstone Presentation Script & Clinical-Technical Manual</title>
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

print("Saved HTML file:", HTML_PATH)

# Convert to PDF via headless Microsoft Edge
edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_exe):
    edge_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

print("Compiling PDF with headless browser:", edge_exe)

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
    
    # Copy to artifact directory
    import shutil
    shutil.copy2(PDF_PATH, ARTIFACT_PDF_PATH)
    print(f"Copied PDF to artifact directory: {ARTIFACT_PDF_PATH}")
else:
    print("FAILED: PDF was not generated.")
