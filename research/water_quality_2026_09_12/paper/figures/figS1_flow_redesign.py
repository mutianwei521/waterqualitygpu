"""Icon-led Figure S1: fixed hydraulics, recording, replay and verification.

All glyphs are authored in matplotlib. No experimental data are invented;
network geometry, matrices, curves and application icons are schematic.
The sole numerical result is the documented CPU replication check (91 cases).
Run directly, or via figS_si.fig_s1().
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Ellipse, Polygon
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from pypdf import PdfReader
import fitz

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
W, H = 176, 146
INK, GREY, LIGHT = '#30363D', '#707A84', '#D6DCE1'
CYAN, TEAL, GOLD = '#56B4E9', '#009E73', '#E69F00'
VERM, PURPLE, PINK = '#D55E00', '#7B61A8', '#CC79A7'
NAVY = '#2F5D7C'
DARK_CYAN, DARK_TEAL, DARK_GOLD = '#247EA8', '#007A59', '#A96E00'
DARK_VERM, DARK_PURPLE, DARK_PINK = '#B84700', '#654A91', '#A94F7D'
PALE_CYAN, PALE_GOLD, PALE_PURPLE = '#E6F5FB', '#FFF1D2', '#F0E7F6'
PALE_TEAL, PALE_PINK, PALE_VERM = '#E2F4ED', '#FAE7F2', '#FCE9DE'
WHITE = '#FFFFFF'


def make_figure():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.6,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'text.color': INK, 'mathtext.fontset': 'dejavusans',
                         'savefig.facecolor': WHITE})
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal'); ax.axis('off')
    texts = []

    def txt(x, y, s, fs=7.6, color=INK, weight='normal', ha='center', va='center', zone=''):
        t = ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight,
                    ha=ha, va=va, linespacing=1.18, zorder=8)
        texts.append((t, zone))
        return t

    def box(x, y, w, h, fc=WHITE, ec=LIGHT, lw=.7, radius=1.2):
        p = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1)
        ax.add_patch(p)
        return p

    def line(xy, color=INK, lw=.9, dash=False, z=3):
        p = Line2D(*zip(*xy), color=color, linewidth=lw,
                   linestyle=(0, (3, 2)) if dash else '-',
                   solid_capstyle='round', zorder=z)
        ax.add_line(p)

    def arrow(xy, color=INK, lw=1, dash=False):
        if len(xy)>2: line(xy[:-1], color, lw, dash)
        ax.annotate('', xy=xy[-1], xytext=xy[-2],
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                    linestyle=(0,(3,2)) if dash else '-',
                                    shrinkA=0, shrinkB=0, mutation_scale=6.7), zorder=4)

    def title(x, y, letter, s, color=INK):
        txt(x, y, letter, 9.2, weight='bold', ha='left', zone='titles')
        txt(x+5, y, s, 8.5, color=color, weight='bold', ha='left', zone='titles')

    def panel(x, y, w, h, fc, ec, accents):
        """Rounded panel with a compact multicolour semantic rail."""
        p = box(x, y, w, h, fc, ec, .9, 1.25)
        usable = w - 2.4
        weights = np.asarray([1.0 + .18*i for i in range(len(accents))])
        weights /= weights.sum()
        x0 = x + 1.2
        for k, (c, frac) in enumerate(zip(accents, weights)):
            ww = usable * frac
            ax.add_patch(Rectangle((x0, y+1.15), ww, 1.15,
                                   facecolor=c, edgecolor='none', zorder=2,
                                   alpha=.96))
            x0 += ww
        return p

    def chip(x, y, w=12, h=10, color=PURPLE, label='GPU'):
        for q in np.linspace(2, w-2, 4):
            line([(x+q,y-1),(x+q,y)],GOLD,.9)
            line([(x+q,y+h),(x+q,y+h+1)],GOLD,.9)
        for q in np.linspace(2,h-2,3):
            line([(x-1,y+q),(x,y+q)],GOLD,.9)
            line([(x+w,y+q),(x+w+1,y+q)],GOLD,.9)
        box(x,y,w,h,color,color,.7,.8)
        box(x+1.4,y+1.4,w-2.8,h-2.8,WHITE,WHITE,0,.3)
        txt(x+w/2,y+h/2,label,7.4,color,weight='bold',zone='chip')

    def matrix(x, y, w, h, rows=4, cols=4, color=PURPLE, dense=False,
               onehot=False, multi=False):
        box(x,y,w,h,WHITE,color,.65,.5)
        for i in range(rows):
            for j in range(cols):
                if (j == i if onehot else dense or (i*3+j*2)%5 < 2):
                    palette=[CYAN,TEAL,GOLD,PURPLE,PINK]
                    c=palette[(i*cols+j)%len(palette)] if (dense or multi) else color
                    ax.add_patch(Rectangle((x+.8+j*(w-1.6)/cols,y+.8+i*(h-1.6)/rows),
                                           (w-2.2)/cols,(h-2.2)/rows,
                                           facecolor=c,edgecolor='none',alpha=.85,zorder=2))

    def chart(x, y, w, h, colors=(TEAL,GOLD,PURPLE)):
        line([(x,y),(x,y+h),(x+w,y+h)],GREY,.65)
        t=np.linspace(0,1,30)
        for k,c in enumerate(colors):
            vals=.15+.20*k+.26*(1-np.exp(-t*3))
            line(list(zip(x+w*t,y+h*vals)),c,.85)

    def check(x,y,color=TEAL,r=2.2):
        ax.add_patch(Circle((x,y),r,facecolor=color,edgecolor='none',zorder=3))
        line([(x-r*.52,y),(x-r*.1,y+r*.42),(x+r*.60,y-r*.45)],WHITE,1,z=4)

    # a | shared binary hydraulic trajectory and operation-level replication.
    title(4,6,'a','Share one hydraulic trajectory',NAVY)
    panel(4,12,78,40,PALE_CYAN,CYAN,(CYAN,TEAL,NAVY))
    txt(43,16.2,'OWA-EPANET Toolkit 2.3',7.8,NAVY,'bold',zone='a')
    nodes=np.array([[11,33],[18,23],[28,26],[27,39],[15,42]])
    for k,(i,j) in enumerate([(0,1),(1,2),(2,3),(3,4),(4,0),(0,3),(1,3)]):
        line([nodes[i],nodes[j]], [NAVY,GREY,TEAL,NAVY,GREY,CYAN,GREY][k], 1.1)
    for x,y in nodes:ax.add_patch(Circle((x,y),.85,facecolor=WHITE,edgecolor=GREY,lw=.7,zorder=4))
    line([(6,27),(6,32),(10,32),(10,27)],INK,.8)
    ax.add_patch(Rectangle((6.3,29),3.4,2.7,facecolor=CYAN,edgecolor='none',zorder=2))
    line([(10,31),(11,33)],GREY,.8)
    box(25.6,20.2,3.7,5.1,PALE_TEAL,TEAL,.7,.2)
    ax.add_patch(Ellipse((27.45,20.2),3.7,1.2,facecolor=WHITE,edgecolor=TEAL,lw=.65,zorder=4))
    arrow([(30,30),(35,30)],CYAN)
    box(35,22,13,18,'#F8FCFE',CYAN,.9,.5)
    line([(38,25),(45,25)],CYAN,.65)
    line([(38,27),(45,27)],GOLD,.65)
    txt(41.5,32,r'$Q,V$',8.2,CYAN,zone='a')
    txt(41.5,43.1,'stored',7.4,GREY,zone='a')
    arrow([(48,29),(52,29),(52,24.5),(56,24.5)],CYAN,.85)
    arrow([(52,29),(52,38),(56,38)],CYAN,.85)
    box(56,20,23,9,'#EDF4F7',NAVY,.8)
    txt(67.5,24.5,'Library quality',7.4,NAVY,weight='bold',zone='a')
    box(56,33.5,23,9,PALE_GOLD,GOLD,.7)
    txt(67.5,38,'Replicator',7.6,GOLD,'bold',zone='a')
    line([(67.5,29.8),(67.5,32.7)],TEAL,.8,True)
    check(51,47.3,r=1.55)
    txt(65,47.2,r'91 cases: $\Delta=0$',7.6,TEAL,zone='a')
    arrow([(79,38),(89.7,38)],GOLD,1.1)

    # b | immutable records, additive inputs, and two tolerances.
    title(90,6,'b','Record immutable dependencies',DARK_GOLD)
    panel(90,12,82,40,PALE_GOLD,GOLD,(GOLD,PINK,TEAL))
    txt(129.5,16.2,'Every write creates a new version',7.6,DARK_GOLD,'bold',zone='b')
    for y,s,c,fc in [(22,'v1',CYAN,PALE_CYAN),(31,'v2',PINK,PALE_PINK)]:
        tc = DARK_CYAN if c == CYAN else DARK_PINK
        box(95,y,10,5,fc,c,.8,.5);txt(100,y+2.5,s,7.5,tc,weight='bold',zone='b')
    arrow([(105,24.5),(113,29)],CYAN,.85)
    arrow([(105,33.5),(113,29)],PINK,.85)
    box(113,26,11,6,PALE_TEAL,TEAL,.9,.5)
    txt(118.5,29,'v3',7.7,DARK_TEAL,'bold',zone='b')
    txt(114,37.5,'weighted assignment',7.4,GREY,zone='b')
    arrow([(124,29),(135,29)],TEAL,1)
    for j,c in [(2,PINK),(1,GOLD)]:
        box(137+j*1.3,22-j*1.2,22,13,WHITE,c,.65,.5)
    matrix(137,22,22,13,3,5,TEAL,multi=True)
    txt(149,37.5,'sparse schedule',7.4,GREY,zone='b')
    box(93,42,35,8,'#FFF9E9',GOLD,.7,.6)
    txt(110.5,44.2,r'$C_{\rm tol}=$ input',7.8,GOLD,zone='b')
    txt(110.5,47.9,'nominal run',7.3,GREY,zone='b')
    box(133,42,36,8,'#F1FAF6',TEAL,.7,.6)
    txt(151,44.2,r'$C_{\rm tol}=0$',7.8,TEAL,zone='b')
    txt(151,47.9,'no parcel merging',7.3,GREY,zone='b')
    txt(132,56.1,'Initial qualities + source slots + constants',7.3,GREY,zone='b')
    # Explicit downstream path without a false gradient through hydraulics.
    arrow([(160,29),(174,29),(174,59),(86,59),(86,69)],GOLD,.9)

    # c | state columns, quality-step fusion and execution routes.
    title(4,62.5,'c','Fuse operations and replay batches',PURPLE)
    panel(4,69,88,43,PALE_PURPLE,PURPLE,(PURPLE,CYAN,GOLD,PINK))
    txt(26,73.3,'Fuse within each step',7.7,PURPLE,'bold',zone='c')
    for x,tag,c in [(9,'W1',CYAN),(23,'W2',GOLD),(46,'W3',PINK)]:
        matrix(x,78,9,7,3,3,c)
        tc = {CYAN:DARK_CYAN, GOLD:DARK_GOLD, PINK:DARK_PINK}[c]
        txt(x+4.5,88.6,tag,7.3,tc,weight='bold',zone='c')
    arrow([(18.5,81.5),(22,81.5)],NAVY,.8)
    arrow([(32.7,81.5),(36.5,81.5)],NAVY,.8)
    line([(38.6,77.8),(38.6,85.2)],VERM,1.3,True)
    arrow([(40.7,81.5),(45.2,81.5)],NAVY,.8)
    txt(68.5,80.8,'cut at set-point\nterms',7.4,VERM,zone='c')
    arrow([(53.5,85.7),(53.5,92.5)],PURPLE,.85)
    matrix(8,94,14,10,4,4,PURPLE,True)
    txt(15,108.1,r'$M\!\times\!B$ state',7.8,PURPLE,zone='c')
    arrow([(23,99),(29,99)],PURPLE,.85)
    box(30,94,14,10,WHITE,GOLD,.65,.7)
    txt(37,98.3,r'$\rho(\theta)$',8,GOLD,zone='c')
    txt(37,107.4,'read factors',7.3,GREY,zone='c')
    arrow([(45,99),(49.5,99)],GOLD,.85)
    chip(51,94,13,10,PURPLE,'FP64')
    arrow([(65.5,99),(73,99)],TEAL,.9)
    chart(75,94,12,10)
    txt(81,108.1,r'$c(t;\theta_b)$',7.9,TEAL,zone='c')
    txt(51,115.9,'Gather / SpMM*  ·  GPU graph replay',7.3,PURPLE,zone='c')
    # Forward concentrations go to either scalar-loss or Jacobian-row seeds.
    arrow([(92,99),(97,99),(97,86),(102,86)],TEAL,.95)

    # d | both seed types use the reverse operator sequence.
    title(101,62.5,'d','Reverse the recorded operations',DARK_VERM)
    panel(101,69,71,43,PALE_VERM,VERM,(VERM,PINK,PURPLE,GOLD))
    txt(118.5,74,'Loss seed',7.7,DARK_VERM,'bold',zone='d')
    txt(153.5,74,'Jacobian rows',7.7,DARK_PINK,'bold',zone='d')
    box(109,78,19,8,'#FFF7F2',VERM,.75,.7)
    txt(118.5,82,r'$\partial L/\partial c$',8.1,VERM,zone='d')
    matrix(146,78,14,8,3,4,PINK,onehot=True)
    txt(166,82,r'$e_r$',8,PINK,zone='d')
    arrow([(118.5,86.8),(118.5,90),(161,90),(161,98.5),(157.5,98.5)],VERM,.85,True)
    arrow([(153,86.8),(153,90)],PINK,.85,True)
    # Right-to-left order is the adjoint sweep, not an implicit hydraulic solve.
    for x in [149,135,121]:matrix(x,95,8,7,3,3,VERM)
    arrow([(148,98.5),(144,98.5)],VERM,.9,True)
    arrow([(134,98.5),(130,98.5)],VERM,.9,True)
    arrow([(120,98.5),(113,98.5)],VERM,.9,True)
    txt(108.5,98.5,r'$\nabla$',10.2,VERM,zone='d')
    x0=107.4
    for w,label,c,fc in [(9,r'$K_b$',TEAL,PALE_TEAL),
                         (9,r'$K_w$',GOLD,PALE_GOLD),
                         (17,r'$K_{b,\rm tank}$',PURPLE,PALE_PURPLE),
                         (9,r'$c_0$',CYAN,PALE_CYAN),
                         (8,r'$s$',PINK,PALE_PINK)]:
        box(x0,104.7,w,5.1,fc,c,.65,.65)
        tc = {TEAL:DARK_TEAL, GOLD:DARK_GOLD, PURPLE:DARK_PURPLE,
              CYAN:DARK_CYAN, PINK:DARK_PINK}[c]
        txt(x0+w/2,107.25,label,7.8,tc,weight='bold',zone='d')
        x0 += w+1.1
    txt(137,115.9,'*SpMM: shared factors + sufficient work',7.3,GREY,zone='d')
    arrow([(108.5,102),(104.5,102),(104.5,118.5),(2,118.5),(2,131),(6.8,131)],VERM,.9,True)

    # e | icons for use and selected re-insertion checks, rather than a claim
    # that every inferred source candidate or parameter vector was re-run.
    title(4,122,'e','Use gradients and check reported solutions in the library',TEAL)
    # Give the final application-and-verification stage enough breathing room; its scope caveat is already
    # stated in the caption, so the enlarged frame can use the full remaining canvas height.
    panel(4,125,168,20.5,'#F3F8F7',TEAL,(TEAL,GOLD,PINK,PURPLE,CYAN))
    # Four application glyphs.
    chart(8,129.5,10,7,(TEAL,))
    for x,y in [(10,131.5),(13,132.7),(17,133.7)]:
        ax.add_patch(Circle((x,y),.55,facecolor=WHITE,edgecolor=TEAL,lw=.55,zorder=3))
    txt(13,141,'Fit',7.4,TEAL,zone='e')
    box(32,130.5,6,6,'#FFF8E8',GOLD,.75,.4)
    line([(35,128.5),(35,130.5),(39.5,130.5)],GOLD,.8)
    ax.add_patch(Polygon([(40,131.5),(38.6,134),(41.4,134)],fc=GOLD,ec='none',zorder=3))
    txt(36,141,'Dose',7.4,GOLD,zone='e')
    ax.add_patch(Circle((59,132.7),3,facecolor=WHITE,edgecolor=PINK,lw=.85,zorder=2))
    ax.add_patch(Circle((59,132.7),1.3,facecolor=PINK,edgecolor='none',zorder=2))
    line([(61.3,135),(64,137.5)],PINK,1.2)
    txt(60,141,'Locate',7.4,PINK,zone='e')
    for (x,y),c in zip([(79,134.5),(85,131.5),(88,136.5)],[PURPLE,CYAN,GOLD]):
        ax.add_patch(Circle((x,y),.7,facecolor=c,edgecolor='none',zorder=2))
    line([(79,134.5),(85,131.5),(88,136.5)],PURPLE,.8)
    ax.add_patch(Polygon([(84,128.3),(82.8,130.5),(85.2,130.5)],fc=WHITE,ec=PURPLE,lw=.6,zorder=3))
    txt(83.5,141,'Sense',7.4,PURPLE,zone='e')
    arrow([(96,133.5),(107.5,133.5)],TEAL,.9)
    # API gateway: a central interface hub connected to three coloured endpoints, with a compact label.
    box(109,128.5,14,11,PALE_CYAN,NAVY,.8,.6)
    hub=(116,132.5)
    for px,py,c in [(112,131,CYAN),(112,134,TEAL),(120,132.5,GOLD)]:
        line([hub,(px,py)],NAVY,.65)
        ax.add_patch(Circle((px,py),.62,facecolor=c,edgecolor=WHITE,lw=.35,zorder=4))
    ax.add_patch(Circle(hub,1.05,facecolor=NAVY,edgecolor=WHITE,lw=.4,zorder=4))
    txt(116,137.2,'API',6.8,NAVY,'bold',zone='e')
    arrow([(124,133.5),(129,133.5)],GOLD,.9)
    chip(132,128.5,17,10,PURPLE,'Library')
    arrow([(151,133.5),(159,133.5)],TEAL,.9)
    check(165,133.5,r=2.65)
    txt(137,142.6,'write back · re-simulate · compare',7.3,NAVY,zone='e')

    return fig, ax, texts


def export():
    OUT.mkdir(exist_ok=True)
    fig, ax, texts = make_figure()
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    canvas=fig.bbox
    outside=[]
    outside_artists=[]
    overlaps=[]
    for t,zone in texts:
        bb=t.get_window_extent(renderer)
        if bb.x0<-.2 or bb.y0<-.2 or bb.x1>canvas.width+.2 or bb.y1>canvas.height+.2:
            outside.append(t.get_text())
    for artist in [*ax.patches, *ax.lines, *ax.texts]:
        bb=artist.get_window_extent(renderer)
        if bb.x0 < -.3 or bb.y0 < -.3 or bb.x1 > canvas.width+.3 or bb.y1 > canvas.height+.3:
            outside_artists.append(type(artist).__name__)
    for i,(ta,za) in enumerate(texts):
        if not za:continue
        aa=ta.get_window_extent(renderer)
        for tb,zb in texts[i+1:]:
            if za!=zb:continue
            bb=tb.get_window_extent(renderer)
            dx=min(aa.x1,bb.x1)-max(aa.x0,bb.x0)
            dy=min(aa.y1,bb.y1)-max(aa.y0,bb.y0)
            if dx>.7 and dy>.7:overlaps.append([ta.get_text(),tb.get_text()])
    for ext in ['svg','pdf','png']:
        fig.savefig(OUT/f'figS1_flow.{ext}',dpi=600)
    import tifffile
    with Image.open(OUT/'figS1_flow.png') as im:
        tifffile.imwrite(OUT/'figS1_flow.tif',np.asarray(im.convert('RGB')),
                         compression='zlib',photometric='rgb',resolution=(600,600),
                         resolutionunit='INCH')
    fig.savefig(OUT/'figS1_flow_preview.png',dpi=220)
    plt.close(fig)
    pdf=PdfReader(OUT/'figS1_flow.pdf')
    page=pdf.pages[0]
    media=[float(page.mediabox.width),float(page.mediabox.height)]
    fonts=page['/Resources']['/Font'].get_object()
    font_types={str(k):str(v.get_object().get('/Subtype')) for k,v in fonts.items()}
    svgroot=ET.parse(OUT/'figS1_flow.svg').getroot()
    svgtexts=svgroot.findall('.//{http://www.w3.org/2000/svg}text')
    doc=fitz.open(OUT/'figS1_flow.pdf')
    pdfpage=doc[0]
    spans=[s for b in pdfpage.get_text('dict')['blocks'] if 'lines' in b
           for ln in b['lines'] for s in ln['spans']]
    glyph_sizes=[s['size'] for s in spans]
    with Image.open(OUT/'figS1_flow.tif') as image:
        tiff_size=list(image.size)
        tiff_dpi=[float(x) for x in image.info['dpi']]
    qa={'size_mm':[W,H], 'pdf_size_pt':media,
        'minimum_base_font_pt':min(t.get_fontsize() for t,z in texts),
        'minimum_pdf_glyph_pt':min(glyph_sizes),
        'pdf_font_types':font_types,'svg_editable_text_elements':len(svgtexts),
        'outside_canvas_text':outside,'outside_canvas_artists':outside_artists,
        'same_region_text_overlap':overlaps,
        'tiff_pixels':tiff_size,'tiff_dpi':tiff_dpi,
        'arxiv_scale':min(165.1/W,137.16/H),
        'arxiv_minimum_glyph_pt':min(glyph_sizes)*min(165.1/W,137.16/H),
        'source':'Methods S1–S5; Figure S1 caption; Figure 1 v2 semantic palette',
        'schematic_only':True,'measured_result':'91 CPU replication cases, zero difference'}
    (OUT/'figS1_flow_qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(qa,indent=2,ensure_ascii=False))
    assert not outside and not outside_artists and not overlaps
    assert min(glyph_sizes) >= 4.9 and qa['arxiv_minimum_glyph_pt'] >= 4.9
    assert len(svgtexts) >= 40 and '/Type3' not in font_types.values()
    assert tiff_dpi == [600.,600.]
    return qa


if __name__=='__main__':
    export()
