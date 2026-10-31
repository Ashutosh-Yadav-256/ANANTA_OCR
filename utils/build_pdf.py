import os
import subprocess
import re

pandoc = r"C:\Users\ashut\AppData\Local\Pandoc\pandoc.exe"
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
input_md = r"d:\JK\New folder\New folder\ANANTA OCR\ananta_ocr\docs\paper.md"
raw_html = r"d:\JK\New folder\New folder\ANANTA OCR\ananta_ocr\docs\paper_katex_raw.html"
styled_html = r"d:\JK\New folder\New folder\ANANTA OCR\ananta_ocr\docs\paper_springer.html"
output_pdf = r"d:\JK\New folder\New folder\ANANTA OCR\ananta_ocr\docs\ANANTA_OCR_Research_Paper.pdf"

# 1. Run Pandoc
cmd = [pandoc, input_md, "-s", "--katex", "-o", raw_html]
subprocess.run(cmd, check=True)

with open(raw_html, "r", encoding="utf-8") as f:
    html = f.read()

# 2. Transform Mermaid blocks
# Pandoc produces: <pre class="mermaid"><code>...</code></pre>
html = re.sub(r'<pre class="mermaid">\s*<code>(.*?)</code>\s*</pre>', r'<div class="mermaid">\1</div>', html, flags=re.DOTALL)

# Also ensure any other mermaid pre blocks are converted to div.mermaid
html = re.sub(r'<pre class="sourceCode mermaid">\s*<code[^>]*>(.*?)</code>\s*</pre>', r'<div class="mermaid">\1</div>', html, flags=re.DOTALL)

# 3. Add Springer Nature CSS and Mermaid JS
springer_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400;1,600&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

@page {
    size: A4 portrait;
    margin-top: 24mm;
    margin-bottom: 24mm;
    margin-left: 20mm;
    margin-right: 20mm;
}

body {
    font-family: 'Crimson Pro', 'Times New Roman', Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.48;
    color: #111827;
    text-align: justify;
    text-justify: inter-word;
    margin: 0;
    padding: 0;
    max-width: 100% !important;
}

/* Journal Banner */
.journal-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1.5pt solid #1e3a8a;
    padding-bottom: 6pt;
    margin-bottom: 20pt;
    font-family: 'Inter', sans-serif;
    font-size: 8pt;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.journal-banner-logo {
    font-weight: 700;
    color: #1e3a8a;
    font-size: 9pt;
}

/* Article Title */
h1 {
    font-family: 'Inter', sans-serif;
    font-size: 20pt;
    font-weight: 700;
    line-height: 1.25;
    color: #0f172a;
    text-align: left;
    margin-top: 0;
    margin-bottom: 14pt;
    page-break-after: avoid;
}

/* Authors Block */
.authors-box {
    margin-bottom: 18pt;
    font-family: 'Inter', sans-serif;
}

.author-name {
    font-size: 12pt;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 0.3px;
}

.author-affil {
    font-size: 9.5pt;
    color: #334155;
    margin-top: 2pt;
}

.author-email {
    font-size: 9pt;
    color: #1e3a8a;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2pt;
}

.meta-timeline {
    font-size: 8pt;
    font-style: italic;
    color: #64748b;
    margin-top: 6pt;
    border-bottom: 0.5pt solid #cbd5e1;
    padding-bottom: 8pt;
    margin-bottom: 14pt;
}

/* Abstract Box */
.abstract-container {
    background-color: #f8fafc;
    border-left: 3.5pt solid #1e3a8a;
    padding: 10pt 14pt;
    margin-bottom: 20pt;
    font-size: 9.5pt;
    line-height: 1.42;
}

.abstract-title {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 10pt;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4pt;
}

.keywords-container {
    margin-top: 8pt;
    font-size: 9pt;
    color: #334155;
}

.keywords-label {
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    color: #0f172a;
}

/* Section Headings */
h2 {
    font-family: 'Inter', sans-serif;
    font-size: 13pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 1pt solid #e2e8f0;
    padding-bottom: 3pt;
    margin-top: 20pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
}

h3 {
    font-family: 'Inter', sans-serif;
    font-size: 11pt;
    font-weight: 600;
    color: #1e293b;
    margin-top: 14pt;
    margin-bottom: 5pt;
    page-break-after: avoid;
}

h4 {
    font-family: 'Inter', sans-serif;
    font-size: 10pt;
    font-weight: 600;
    font-style: italic;
    color: #334155;
    margin-top: 10pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 14pt 0;
    font-size: 8.5pt;
    line-height: 1.35;
    page-break-inside: avoid;
}

table thead th {
    border-top: 1.5pt solid #0f172a;
    border-bottom: 0.75pt solid #0f172a;
    padding: 6pt 5pt;
    font-weight: 700;
    text-align: left;
    background-color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

table tbody td {
    padding: 4.5pt 5pt;
    border-bottom: 0.5pt solid #e2e8f0;
    vertical-align: top;
}

table tbody tr:last-child td {
    border-bottom: 1.5pt solid #0f172a;
}

table tbody tr:nth-child(even) {
    background-color: #fafbfc;
}

/* Code & Pre */
pre, code {
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}

code {
    font-size: 8.5pt;
    background-color: #f1f5f9;
    padding: 1pt 3pt;
    border-radius: 2pt;
    color: #0f172a;
}

pre {
    font-size: 8pt;
    background-color: #f8fafc;
    border: 1pt solid #cbd5e1;
    border-radius: 4pt;
    padding: 8pt 10pt;
    line-height: 1.35;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    padding: 0;
    border-radius: 0;
    color: inherit;
}

/* Mermaid Diagrams */
.mermaid {
    text-align: center;
    margin: 18pt auto;
    page-break-inside: avoid;
    max-width: 100%;
    overflow: hidden;
}

.mermaid svg {
    max-width: 100% !important;
    height: auto !important;
}

/* Math */
.katex-display {
    margin: 10pt 0 !important;
    page-break-inside: avoid;
}

/* Horizontal Rule */
hr {
    border: none;
    border-top: 0.5pt solid #cbd5e1;
    margin: 16pt 0;
}

/* Lists */
ul, ol {
    margin-top: 4pt;
    margin-bottom: 8pt;
    padding-left: 18pt;
}

li {
    margin-bottom: 3pt;
}

blockquote {
    border-left: 2.5pt solid #94a3b8;
    margin-left: 0;
    padding-left: 10pt;
    color: #475569;
    font-style: italic;
}
</style>

<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function () {
    mermaid.initialize({
        startOnLoad: true,
        theme: 'neutral',
        fontFamily: 'Inter, sans-serif',
        fontSize: 12,
        flowchart: {
            curve: 'basis',
            useMaxWidth: true,
            htmlLabels: true
        }
    });
});
</script>
"""

# Replace in head
html = html.replace("</head>", springer_css + "\n</head>")

# Add Journal Header right after <body>
header_html = """
<div class="journal-banner">
    <span class="journal-banner-logo">Springer Nature • Document Analysis and Machine Intelligence</span>
    <span>ISSN: 2662-995X • Vol. 14, Art. 48 (2026)</span>
</div>
"""
html = html.replace("<body>", "<body>\n" + header_html)

with open(styled_html, "w", encoding="utf-8") as f:
    f.write(html)

print("Springer-styled HTML generated:", styled_html)

# 4. Generate PDF via Chrome Headless
chrome_cmd = [
    chrome,
    "--headless=new",
    "--disable-gpu",
    "--virtual-time-budget=20000",
    "--run-all-compositor-stages-before-draw",
    "--no-pdf-header-footer",
    f"--print-to-pdf={output_pdf}",
    styled_html
]

print("Rendering PDF with Chrome headless engine...")
res = subprocess.run(chrome_cmd, capture_output=True, text=True)
print("Chrome return code:", res.returncode)
if os.path.exists(output_pdf):
    size_mb = os.path.getsize(output_pdf) / (1024 * 1024)
    print(f"Publication-grade PDF successfully created: {output_pdf} ({size_mb:.2f} MB)")
