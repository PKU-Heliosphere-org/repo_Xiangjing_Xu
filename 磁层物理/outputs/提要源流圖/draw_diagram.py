from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import zipfile, re, json
from lxml import etree as ET

OUT = Path(__file__).parent
S = 600 / 25.4
W, H = round(210*S), round(297*S)
im = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(im)
INK = '#203C4B'
LINE = '#426779'
BORDER = '#7896A5'
FILL = '#F5F9FB'
ACCENT = '#8A6232'
NOTE = '#FCF7EE'
fontpath = 'C:/Windows/Fonts/simsun.ttc'
numberpath = 'C:/Windows/Fonts/times.ttf'
labels = []

def xy(p): return tuple(round(v*S) for v in p)
def runs(t, size):
    return [(s, ImageFont.truetype(numberpath if re.fullmatch(r'[0-9.\-]+',s) else fontpath,round(size*S)))
            for s in re.findall(r'[0-9.\-]+|[^0-9.\-]+',t)]

def text(cx, cy, lines, size=3.75, color=INK, bold=False, spacing=1.4):
    step = size*spacing
    for i, t in enumerate(lines):
        yy = cy+(i-(len(lines)-1)/2)*step
        rr=runs(t,size)
        xx=cx*S-sum(d.textlength(s,font=f) for s,f in rr)/2
        # Share the typographic baseline across Song and Times New Roman.
        baseline=(yy+size*.36)*S
        for s,f in rr:
            d.text((round(xx),round(baseline)),s,font=f,fill=color,anchor='ls')
            xx+=d.textlength(s,font=f)

def side_label(cx,cy,t,size=3.65):
    rr=runs(t,size)
    lengths=[d.textlength(s,font=f) for s,f in rr]
    label=Image.new('RGBA',(round(sum(lengths)+4*S),round(7*S)),(255,255,255,0))
    ld=ImageDraw.Draw(label)
    xx=2*S
    for (s,f),length in zip(rr,lengths):
        ld.text((round(xx),round((3.5+size*.36)*S)),s,font=f,fill=INK,anchor='ls')
        xx+=length
    label=label.rotate(90,expand=True)
    im.paste(label,(round(cx*S-label.width/2),round(cy*S-label.height/2)),label)

def path(points, color=LINE, width=.38, arrow=True, dashed=False):
    if dashed:
        import math
        for a,b in zip(points,points[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dy)
            for k in range(0, int(length*10), 24):
                t=k/10
                e=min(t+1.3,length)
                d.line([xy((a[0]+dx*t/length,a[1]+dy*t/length)),xy((a[0]+dx*e/length,a[1]+dy*e/length))],fill=color,width=round(width*S))
    else:
        d.line([xy(p) for p in points], fill=color,width=round(width*S),joint='curve')
    if arrow:
        import math
        a,b=points[-2:]
        dx,dy=b[0]-a[0],b[1]-a[1]
        ll=math.hypot(dx,dy); ux,uy=dx/ll,dy/ll
        z=1.8; r=.85
        d.polygon([xy(b),xy((b[0]-z*ux+r*uy,b[1]-z*uy-r*ux)),xy((b[0]-z*ux-r*uy,b[1]-z*uy+r*ux))],fill=color)

def node(x,y,w,h,lines,size=3.75,note=False,bold=False):
    labels.append(''.join(lines))
    bounds=xy((x,y,x+w,y+h))
    if note:
        d.rectangle(bounds,fill=NOTE,outline=ACCENT,width=round(.4*S))
    else:
        d.rounded_rectangle(bounds,radius=round(2*S),fill=FILL,outline=BORDER,width=round(.36*S))
    assert all(sum(d.textlength(s,font=f) for s,f in runs(t,size)) <= (w-4)*S for t in lines), lines
    text(x+w/2,y+h/2,lines,size,ACCENT if note else INK,bold)

# All source connections, rearranged into a portrait layout.
path([(105,44),(105,59)])
path([(105,75),(105,91)])
path([(105,109),(105,129)])
path([(105,147),(105,164)])
path([(105,181),(105,190)])
path([(105,216),(105,274)])
# Early manuscript branch.
path([(77,100),(36,100),(36,129)])
path([(36,147),(36,274)])
# The three early book-front abstracts share the 43-stage source.
path([(133,100),(137,100),(137,189)],arrow=False)
for yy in (120,154,189):
    path([(137,yy),(141,yy)])
# Later copies derive from the stage after the second presentation.
path([(105,219),(164.5,219),(164.5,224)])
# Wenlan is later than Wensu: its branch is below the entire Wensu node.
path([(105,248),(73,248),(73,252)])
# Wenlan copy is used to revise the line leading to the Zhejiang edition.
path([(48,260.5),(36,260.5)])
# Revision of the three book-front abstracts, 51–53.
path([(105,261),(193,261),(193,120)],arrow=False)
for yy in (120,154,189):
    path([(193,yy),(188,yy)])
# The first required box belongs before the first presentation.
path([(105,83),(138,83),(138,70),(146,70)],color=ACCENT,width=.3,arrow=False,dashed=True)

text(105,15,['《滇考》提要源流圖'],6.2,bold=True)
node(46,28,118,16,['浙江進呈《滇考》並附簡略提要'],4.2,bold=True)
node(77,59,56,16,['書前提要的初步匯集','（39）'])
node(77,91,56,18,['書前提要的彙編','（43）'])
node(146,59,49,22,['加入對於體例、','內容的評價'],3.9,note=True)
node(11,129,50,18,['南圖稿本'],4)
node(77,129,56,18,['《總目》初次進呈','（46.2）'],3.85,bold=True)
node(77,164,56,17,['《總目》再次進呈','（47.7）'],3.85,bold=True)
node(77,190,56,26,['否定『與記』特徵，','強調『紀事本末之體』'],3.7,note=True)
node(141,111,47,18,['文淵閣書前提要','（45.9）'])
node(141,145,47,18,['文溯閣書前提要','（47.5）'])
node(141,180,47,18,['文津閣書前提要','（49.3）'])
node(141,224,47,20,['文溯閣抄本《總目》','（48.4-48.8）'],3.6)
node(48,252,50,17,['文瀾閣抄本《總目》','（53）'],3.6)
node(11,274,50,17,['浙本《總目》','（59-60）'],3.8)
node(80,274,50,17,['殿本總目','（60）'],3.8)
node(141,274,47,17,['文瀾閣書前提要','（？）'],3.75)
text(42,255.5,['校改'],3.25)
labels.append('校改')
# Label beside precisely the three affected abstracts.
side_label(201,154.5,'覆校（51-53）',3.85)
labels.append('覆校（51-53）')

# Verify every original text object is represented, allowing the user's quote marks.
source = Path('D:/program files/xwechat_files/wxid_4jcdodmx8ean22_26ed/msg/file/2026-09/提要源流图.docx')
with zipfile.ZipFile(source) as z:
    root=ET.fromstring(z.read('word/document.xml'))
    source_labels=[''.join(e.xpath('.//w:t/text()',namespaces=root.nsmap)) for e in root.findall('.//w:txbxContent',root.nsmap)]
def norm(t): return re.sub(r'\s+', '',t).replace('「','『').replace('」','』')
expected_labels=[t.replace('書前提要的初步匯集（43）','書前提要的彙編（43）') for t in source_labels]
assert sorted(map(norm,expected_labels)) == sorted(map(norm,labels)), (expected_labels,labels)
im.save(OUT/'提要源流圖_A4_600dpi_修訂版2.png',dpi=(600,600),optimize=True)
im.resize((1488,2105),Image.Resampling.LANCZOS).save(OUT/'preview.png')
(OUT/'content_audit.json').write_text(json.dumps({'all_text_objects_preserved':True,'text_object_count':len(labels),'image_pixels':[W,H],'dpi':600,'source_text':source_labels,'output_text':labels},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Created {W} x {H} PNG. All {len(labels)} source text objects verified.')
