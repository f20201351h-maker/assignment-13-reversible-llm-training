"""Export the generated Markdown report to portable HTML and a paginated PDF."""
import base64
import html
import re
import hashlib
import importlib.metadata
import json
from pathlib import Path

import markdown
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'REPORT.md').read_text(encoding='utf-8')
body=markdown.markdown(source, extensions=['tables','fenced_code'])
def embed(match):
    target=ROOT/match.group(1)
    if not target.is_file():
        return match.group(0)
    return 'src="data:image/png;base64,'+base64.b64encode(target.read_bytes()).decode()+'"'
body=re.sub(r'src="([^"]+)"',embed,body)
css='''body{font:17px/1.6 system-ui,sans-serif;color:#18232e;max-width:1080px;margin:48px auto;padding:0 28px}
h1{font-size:2.2rem;line-height:1.15}h2{margin-top:2.3rem;color:#153e57}img{width:100%;height:auto}
table{border-collapse:collapse;width:100%;font-size:.84rem}th,td{padding:9px 8px;border-bottom:1px solid #dbe1e7;text-align:left}
th{background:#edf3f7}pre{background:#f2f5f8;padding:18px;overflow:auto;font-size:.83rem}code{font-size:.88em}
a{color:#006a94}p{max-width:90ch}@media print{body{font-size:10pt;margin:0}h2{break-after:avoid}img,table{break-inside:avoid}}'''
(ROOT/'REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reversible LLM Training Study</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')

for name,family,weight in [('Body','DejaVu Sans','normal'),('Bold','DejaVu Sans','bold'),('Mono','DejaVu Sans Mono','normal')]:
    path=font_manager.findfont(font_manager.FontProperties(family=family,weight=weight))
    pdfmetrics.registerFont(TTFont(name,path))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
styles={
    'body':ParagraphStyle('body',fontName='Body',fontSize=9,leading=13,spaceAfter=8),
    'title':ParagraphStyle('title',fontName='Bold',fontSize=22,leading=27,spaceAfter=18),
    'heading':ParagraphStyle('heading',fontName='Bold',fontSize=13,leading=17,spaceBefore=14,spaceAfter=8,keepWithNext=True),
    'cell':ParagraphStyle('cell',fontName='Body',fontSize=6.8,leading=9,alignment=TA_LEFT),
    'code':ParagraphStyle('code',fontName='Mono',fontSize=6.5,leading=9,spaceBefore=5,spaceAfter=10),
}
def inline(text):
    # Protect literal equations/code before interpreting Markdown punctuation.
    parts = re.split(r'(`[^`]+`)', text)
    return ''.join('<font name="Mono">'+html.escape(part[1:-1])+'</font>'
                   if part.startswith('`') and part.endswith('`')
                   else inline_prose(part) for part in parts)

def inline_prose(text):
    text=html.escape(text).replace('–','-').replace('—','-')
    text=re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#006a94">\1</link>',text)
    text=re.sub(r'\[([^]]+)\]\([^)]+\)',r'\1',text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'\*([^*]+)\*',r'<i>\1</i>',text)
    return text

flow=[]
lines=source.splitlines(); index=0
width=A4[0]-92
while index<len(lines):
    line=lines[index].strip()
    if not line:
        index+=1; continue
    if line.startswith('```'):
        chunk=[]; index+=1
        while index<len(lines) and not lines[index].startswith('```'):
            chunk.append(lines[index]); index+=1
        flow.append(Preformatted('\n'.join(chunk),styles['code'],maxLineLength=108))
    elif line.startswith('|'):
        rows=[]
        while index<len(lines) and lines[index].strip().startswith('|'):
            cells=[v.strip() for v in lines[index].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',v) for v in cells):
                rows.append([Paragraph(inline(v),styles['cell']) for v in cells])
            index+=1
        columns=len(rows[0]); widths=[width/columns]*columns
        if columns==7:
            ratios=[1.28,1.30,0.92,0.94,0.65,1.24,1.00]
            widths=[width*r/sum(ratios) for r in ratios]
        tab=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3f7')),
                                ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),
                                ('TOPPADDING',(0,0),(-1,-1),6),
                                ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#dbe1e7'))]))
        flow.extend([tab,Spacer(1,10)]); continue
    elif line.startswith('!['):
        target=ROOT/re.search(r'\]\(([^)]+)\)',line).group(1)
        if not target.exists(): raise FileNotFoundError(target)
        item=Image(str(target)); scale=min(width/item.imageWidth,460/item.imageHeight)
        item.drawWidth=item.imageWidth*scale; item.drawHeight=item.imageHeight*scale
        before = Spacer(1,8)
        before.keepWithNext = True
        flow.extend([before,item,Spacer(1,10)])
    elif line.startswith('# '):
        flow.append(Paragraph(inline(line[2:]),styles['title']))
    elif line.startswith('## '):
        flow.append(Paragraph(inline(line[3:]),styles['heading']))
    else:
        chunk=[line]; index+=1
        while index<len(lines) and lines[index].strip() and not lines[index].startswith(('#','|','```','![','- ')):
            chunk.append(lines[index].strip()); index+=1
        flow.append(Paragraph(inline(' '.join(chunk)),styles['body'])); continue
    index+=1

def footer(canvas,doc):
    canvas.saveState(); canvas.setFont('Body',7)
    canvas.setFillColor(colors.HexColor('#687582'))
    canvas.drawString(46,24,'Reversible LLM training study | research report')
    canvas.drawRightString(A4[0]-46,24,str(doc.page)); canvas.restoreState()
doc=SimpleDocTemplate(str(ROOT/'REPORT.pdf'),pagesize=A4,rightMargin=46,leftMargin=46,
                      topMargin=42,bottomMargin=42,title='Reversible LLM Training Study')
doc.build(flow,onFirstPage=footer,onLaterPages=footer)
manifest={'source_sha256':hashlib.sha256((ROOT/'REPORT.md').read_bytes()).hexdigest(),
          'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'versions':{name:importlib.metadata.version(name) for name in ('Markdown','reportlab','matplotlib')},
          'outputs':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('REPORT.html','REPORT.pdf')}}
(ROOT/'audit/report_export.json').write_text(json.dumps(manifest,indent=2))
print('Exported REPORT.html and REPORT.pdf from REPORT.md')
