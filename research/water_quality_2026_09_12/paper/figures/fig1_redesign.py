"""Figure 1: fixed-hydraulics water-quality recording, replay and adjoints.

Schematic-led composite. All quantitative values come from saved result JSONs.
Panel a is illustrative. c compares two documented verification runs; d retains
all eleven sweeps; e excludes setup; f is the saved prefix-row benchmark.
Python only; editable SVG/PDF, 600-dpi PNG/TIFF, source CSV and provenance manifest.
"""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import paperfig as pf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon, Ellipse
from matplotlib.ticker import LogFormatterMathtext, NullLocator
pf.setup()
mpl.rcParams.update({'axes.grid':False,'axes.linewidth':.55,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
W,H=176,194
BLUE,TEAL,ORANGE='#326C9F','#398B8D','#C37A52'
INK,GREY,LIGHT='#263640','#74818B','#CCD3D9'
NET={'Net1':'#69757F','Net2_react':TEAL,'Jilin_kw':'#B49458','Net3_CHEM_kw_mixed_tanks':'#7C71A5','LTown_kw':BLUE,'BWSN2_kw':TEAL}
NAMES={'Net1':'Net1','Net2_react':'Net2','Jilin_kw':'Jilin','Net3_CHEM_kw_mixed_tanks':'Net3','LTown_kw':'L-Town','BWSN2_kw':'BWSN2'}
DASH,DOT=(0,(3.,1.8)),(0,(1.,1.6))
sources={}
def load(p):
 p=Path(p);sources[p.relative_to(pf.RES).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding='utf-8'))
g1=[r for fn in ['g1_core_nextQ.json','g1_core_stepQ.json','g1_variants_nextQ.json','g1_corpus_nextQ.json'] for r in load(pf.DATA['g1']/fn)['results']]
g2=sorted(load(pf.DATA['g3s']/'g2_fused_spmm.json')['results'],key=lambda x:x['fused_terms'])
fd={r['network']:r for r in load(pf.DATA['g2fd']/'g2_nextQ_ctol0_cuda.json')['results']}
sweeps=[load(p) for p in sorted(pf.DATA['fds'].glob('g2_fd_sweep_*.json'))]
v2=load(pf.DATA['v2']/'g3_bench_v2_cuda.json')['results']
shared={r['network']:r for p in sorted(pf.DATA['g3s'].glob('g3_shared_*.json')) for r in load(p)['results']}
terms=np.array([r['fused_terms'] for r in g2]);forward=np.array([r['forward_rel_vs_dll'] for r in g2]);adjoint=np.array([r['adjoint_vs_autograd']['max_rel_err'] for r in g2])
worst,directions,off=[],[],[]
for r in g2:
 rr=fd.get(r['network'],{});groups=rr.get('fd_dll') or {};k=max(groups,key=lambda k:groups[k]['rel_err']) if groups else None
 worst.append(groups[k]['rel_err'] if k is not None else np.nan);directions.append(k);off.append(not rr.get('reactflag',True))
worst,off=np.array(worst),np.array(off)
fig=plt.figure(figsize=(W/25.4,H/25.4))
def axis(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def ft(x,y,s,size=7.5,weight='normal',color=INK,ha='left',va='center',**kw):return fig.text(x/W,y/H,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,**kw)
def panel(x,y,l,title):ft(x,y,l,9,'bold');ft(x+5,y,title,8,'bold')
def style(ax):
 ax.spines[['top','right']].set_visible(False)
 for s in ['left','bottom']:ax.spines[s].set_color(GREY)
 ax.tick_params(labelsize=7.2,colors=INK,length=2,width=.5);ax.grid(axis='y',which='major',color='#E5EAED',lw=.35,zorder=0);ax.set_axisbelow(True)
def logs(ax):
 ax.set_xscale('log');ax.set_yscale('log');ax.xaxis.set_major_formatter(LogFormatterMathtext());ax.yaxis.set_major_formatter(LogFormatterMathtext())
 ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
def key(x,y,color,label,marker=None,ls='-',size=7):
 fig.add_artist(Line2D([x/W,(x+5)/W],[y/H,y/H],transform=fig.transFigure,color=color,linewidth=.9,linestyle=ls,marker=marker,markersize=3,markerfacecolor='white',markevery=[1]))
 ft(x+6.5,y,label,size,color=color)

# a. Coordinates are millimetres, preserving natural icon proportions.
panel(4,190,'a','One transport record, many parameter sets')
a=axis(4,139,168,47);a.set(xlim=(0,168),ylim=(0,47));a.axis('off')
def text(x,y,s,fs=7.2,color=INK,ha='center',weight='normal',**kw):return a.text(x,y,s,fontsize=fs,color=color,ha=ha,va='center',fontweight=weight,**kw)
def arrow(p,q,color=BLUE,lw=1.,ls='-',scale=7,**kw):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=scale,lw=lw,color=color,linestyle=ls,shrinkA=0,shrinkB=0,**kw))
def box(x,y,w,h,fc='#EDF3F8',ec=LIGHT,lw=.6,r=1.2):a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw))
def reservoir(x,y):
 a.add_patch(Polygon([(x,y+7),(x,y),(x+8,y),(x+8,y+7)],closed=False,fill=False,ec=GREY,lw=1.2))
 a.add_patch(Polygon([(x+.8,y+.6),(x+7.2,y+.6),(x+7.2,y+4.5),(x+6,y+4.9),(x+4.5,y+4.4),(x+3,y+4.9),(x+1.5,y+4.4),(x+.8,y+4.7)],fc='#A9C9DC',ec='none'))
def tank(x,y):
 a.add_patch(Rectangle((x,y),5.5,8,fc='#D8E8F1',ec=GREY,lw=.8));a.add_patch(Rectangle((x+.3,y+.4),4.9,3.6,fc='#94BBD3',ec='none'))
 a.add_patch(Ellipse((x+2.75,y+8),5.5,1.8,fc='white',ec=GREY,lw=.8));a.add_patch(Ellipse((x+2.75,y),5.5,1.8,fc='#94BBD3',ec=GREY,lw=.8))
def chip(x,y,w=12,h=11):
 for xx in np.linspace(x+2,x+w-2,5):a.plot([xx,xx],[y-1.2,y+h+1.2],color='#BA9B64',lw=1.35,zorder=1)
 for yy in np.linspace(y+1.8,y+h-1.8,4):a.plot([x-1.2,x+w+1.2],[yy,yy],color='#BA9B64',lw=1.35,zorder=1)
 box(x,y,w,h,'#576D7B','#385266',.9,.8);box(x+1.8,y+1.8,w-3.6,h-3.6,'#C3D7E5','#C3D7E5',.4,.3);text(x+w/2,y+h/2,'GPU',7.2,'#314C60',weight='bold')
text(23.5,44,'Fixed hydraulics',8,BLUE,weight='bold');text(23.5,39.4,'OWA-EPANET Toolkit 2.3',7,GREY)
nodes={0:(13,27),1:(23,32),2:(36,31),3:(21,19),4:(35,17),5:(12,13)}
for u,v in [(0,1),(1,2),(0,3),(1,3),(2,4),(3,4),(3,5),(5,4)]:
 p,q=nodes[u],nodes[v];a.plot([p[0],q[0]],[p[1],q[1]],color='#758895',lw=1.5,zorder=1)
a.plot([8,13],[27,27],color='#758895',lw=1.5);a.plot([36,41],[31,31],color='#758895',lw=1.5);reservoir(0,23);tank(40,29)
for x,y in nodes.values():a.add_patch(Circle((x,y),.9,fc='white',ec='#5E7483',lw=.8,zorder=3))
box(15,17.3,18,3.7,'white','#8095A3',.6,.3)
for j,co in enumerate(['#BFD9E3','#8EBCCF','#659FB8','#D8E8EE']):a.add_patch(Rectangle((15.4+j*4.3,17.7),4.1,2.9,fc=co,ec='white',lw=.3,zorder=4))
arrow((19,24),(25,24),TEAL,.85);text(26,12.4,'water parcels',7,GREY);text(45.5,24.3,'Q(t), V(t)',7.2,BLUE,ha='right');arrow((46,28),(54,28),BLUE,1.1)
text(80,44,'Record once',8,BLUE,weight='bold');text(80,39.4,'Parcel operations',7,GREY)
for j,co in enumerate(['#9BC2D7','#C8DDE8','#709FBF']):box(57+j*7.5,28.5,6,4,co,'#6F98B0',.45,.35);text(60+j*7.5,30.5,f'c{j+1}',7,'#335E78')
for j in range(3):arrow((60+j*7.5,28.2),(69,22.4),BLUE,.55,scale=5)
box(64.8,17.8,9,4.6,'#D5E5EE',BLUE,.6,.4);text(69.3,20.1,'c4',7,BLUE);arrow((73.8,20),(82.8,20),BLUE,.8)
for offset in [2.3,1.15,0]:box(85+offset,15+offset,14,15,'white','#9FB6C7',.55,.5)
for row,col in [(0,0),(0,3),(1,1),(2,0),(2,2),(3,1),(3,3)]:a.add_patch(Rectangle((87+col*2.7,17+row*2.7),1.9,1.9,fc=BLUE if row==col else '#A2C0D5',ec='none'))
text(79,11,'immutable versions → schedule',7,GREY)
# Fixed operators and new parameters enter the GPU independently.
a.plot([103,108,108,139],[28,28,34.5,34.5],color=BLUE,lw=.85)
arrow((139,34.5),(139,32.1),BLUE,.85,scale=6)
text(142,44,'Replay + differentiate',8,BLUE,weight='bold');text(142,39.4,'First-order reaction factors',7,GREY)
for j in [2,1,0]:box(116+j*1.2,18+j*1.3,9,13,'white','#9DB9CD',.6,.4)
for j in range(3):a.add_patch(Rectangle((117.5+j*2.4,20),1.4,8,fc=['#C2D9E6','#8FB7CF','#6095BB'][j],ec='none'))
text(121,14.8,r'$\theta_1\ldots\theta_B$',7.5,BLUE);arrow((127,25),(131,25),BLUE,.9);chip(133,19.5,12,11);text(139,15.2,'FP64',7,GREY);arrow((146.7,25),(151,25),BLUE,.9)
a.plot([153,153,166],[32,20,20],color='#97A5AF',lw=.5);xx=np.linspace(153.5,165.5,45)
for jj,co in enumerate([BLUE,'#8BAEC5',TEAL]):a.plot(xx,22.2+jj*1.9+1.7*np.exp(-(xx-153.5)/4)+.4*np.sin((xx-153.5)/2+jj),color=co,lw=.8)
text(159.5,34.5,'c(t)',7.5,BLUE);text(159.5,16,'loss L(c)',7,INK);arrow((166.5,18),(166.5,8),BLUE,.9,DASH)
arrow((166.5,7.3),(42,7.3),BLUE,1.,DASH);text(107,5.1,r'reverse pass  $\partial L/\partial\theta$',7.5,BLUE)
box(0,1,40,7,'#EFF5F8','#9EB9CC',.6,.7);text(20,4.4,r'$\theta=\{K_b,K_w,s,c_0\}$',7.7,BLUE)

# b/c. Zero CPU difference is a textual result, not a point on a logarithmic axis.
b=axis(14,83,43,40);c=axis(73,83,43,40)
for ax in (b,c):style(ax);logs(ax);ax.set_xlim(1e4,1e6);ax.set_xticks([1e4,1e5,1e6]);ax.set_xlabel('Recorded terms',fontsize=7.5)
panel(4,134,'b','Forward fidelity');panel(63,134,'c','Gradient checks');ft(14,128,'25 GPU configurations',7,color=GREY)
b.scatter(terms,forward,s=13,facecolors='#9ABED7',edgecolors=BLUE,linewidths=.55,zorder=3);b.set_ylim(5e-16,5e-14);b.set_yticks([1e-15,1e-14]);b.set_ylabel('Relative concentration error',fontsize=7.5)
b.text(.05,.94,r'max 1.7 × 10$^{-14}$',transform=b.transAxes,fontsize=7.5,color=BLUE,va='top');ft(14,74,'CPU: 91 cases, zero difference',7,color=GREY)
key(73,128,BLUE,'Adjoint',marker='o');key(96,128,ORANGE,'FD',marker='^')
c.scatter(terms,adjoint,s=13,facecolors='#9ABED7',edgecolors=BLUE,linewidths=.55,zorder=4);valid=np.isfinite(worst)
c.scatter(terms[valid&~off],worst[valid&~off],s=17,marker='^',facecolors='#E9C2AB',edgecolors=ORANGE,linewidths=.6,zorder=3)
c.scatter(terms[valid&off],worst[valid&off],s=17,marker='^',facecolors='#D6DCE1',edgecolors=GREY,linewidths=.6,zorder=3)
c.set_ylim(5e-17,1e-2);c.set_yticks([1e-16,1e-12,1e-8,1e-4]);c.set_ylabel('Relative gradient error',fontsize=7.5)
c.text(.05,.32,r'adjoint ≤ 3.1 × 10$^{-15}$',transform=c.transAxes,fontsize=7.5,color=BLUE);ft(73,74,'Grey: reactions off at recording',7,color=GREY)

# d. All eleven sweeps, separated into regular/affine and boundary directions.
panel(122,134,'d','FD step-size sensitivity');d=axis(132,106,40,19);d2=axis(132,82,40,17)
for ax in (d,d2):style(ax);logs(ax);ax.set_xlim(5e-8,2e-1);ax.set_xticks([1e-7,1e-4,1e-1])
regular=[];boundary=[]
for dd in sweeps:
 er={x['h']:x['rel_err'] for x in dd['runs']['M1_ctol0']['sweep']}
 (boundary if er[1e-3]>1e-6 and np.log10(er[1e-2]/er[1e-3])<1.5 else regular).append(dd)
def series(dd,mode):rr=dd['runs'][mode]['sweep'];return [x['h'] for x in rr],[x['rel_err'] for x in rr]
for dd in regular:
 for mode,co,ls in [('M1_ctol0',BLUE,'-'),('M0_inp_ctol',ORANGE,DASH)]:d.plot(*series(dd,mode),color=co,ls=ls,lw=.7,alpha=.83,marker='o' if mode=='M1_ctol0' else None,ms=1.6,mew=.3)
d.set_ylim(1e-15,2);d.set_yticks([1e-14,1e-7,1]);d.tick_params(labelbottom=False);key(132,129,BLUE,'Zero',size=7);key(153,129,ORANGE,'Default',ls=DASH,size=7)
ft(132,101.7,'Zero-coefficient directions',7,color=GREY)
hh=np.array([4e-4,2e-2]);d.plot(hh,1e-7*(hh/1e-2)**2,color=GREY,lw=.6);d.text(2.8e-2,1e-7,r'$h^2$',fontsize=7.5,color=GREY)
boundary_marks={('Net2','kb'):'s',('Net2','kw'):'o',('Net2_MIX2','kw'):'D',('Net3_CHEM_mixed_tanks','kw'):'^'}
for dd in boundary:d2.plot(*series(dd,'M1_ctol0'),color=BLUE,lw=.85,marker=boundary_marks.get((dd['case'],dd['param']),'D'),ms=2.4,mfc='white',mec=BLUE,mew=.6)
d2.set_ylim(3e-7,1);d2.set_yticks([1e-6,1e-3,1]);d2.set_xlabel(r'FD step $h$',fontsize=7.5);ft(122.5,111,'Relative gradient error',7.5,ha='center',rotation=90)

# e. Two-column width avoids putting three engines and four networks into a tiny box.
panel(4,66,'e','Batched gradient cost');e=axis(14,13,86,39);style(e);logs(e)
key(14,55.5,GREY,'FD, projected',ls=DASH);key(47,55.5,GREY,'Per-operator',ls=DOT);key(78,55.5,BLUE,'Fused')
ecsv=[]
for j,r in enumerate(v2):
 col=NET[r['network']];key(14+j*23,60.5,col,f"{NAMES[r['network']]} ({r['n_theta']})")
 for eng,ls,lw in [('adjoint',DOT,.85),('fused','-',1.15)]:
  pts=[(x['B'],x['graph_forward_backward_s']/x['B']) for x in r['engines'][eng]['batches'] if 'graph_forward_backward_s' in x]
  e.plot(*zip(*pts),color=col,ls=ls,lw=lw,marker='o',ms=2.6,mfc='white',mew=.6)
  ecsv += [dict(network=r['network'],n_theta=r['n_theta'],engine=eng,B=B,seconds_per_set=t,projected=False) for B,t in pts]
 e.axhline(r['fd_gradient_cost_s'],color=col,ls=DASH,lw=.8);ecsv.append(dict(network=r['network'],n_theta=r['n_theta'],engine='library_fd',B=1,seconds_per_set=r['fd_gradient_cost_s'],projected=True))
e.set_ylim(3e-6,2);e.set_xlim(.8,5500);e.set_xticks([1,16,256,4096]);e.set_xticklabels(['1','16','256','4,096']);e.set_yticks([1e-5,1e-3,1e-1,1]);e.set_ylabel('Time per parameter set (s)',fontsize=7.5);e.set_xlabel(r'Batch size $B$',fontsize=7.5)

# f. The timed prefix rows are retained and described accurately in the manifest/caption.
panel(115,66,'f','Jacobian-row throughput');f=axis(125,13,47,39);style(f);logs(f)
key(125,60.5,BLUE,'L-Town');key(149,60.5,TEAL,'BWSN2');ft(125,55.5,'905 / 14,820 ordinary pipes',7,color=GREY);fcsv=[]
for net in ['LTown_kw','BWSN2_kw']:
 for eng,ls,lw in [('fused_gather',DOT,.8),('fused_spmm',DASH,.85),('fused','-',1.15)]:
  pts=[(x['B'],x['jac_rows']['rows_per_s']) for x in shared[net]['engines'][eng]['batches'] if x.get('jac_rows')]
  f.plot(*zip(*pts),color=NET[net],ls=ls,lw=lw,marker='o',ms=2.4,mfc='white',mew=.5);fcsv += [dict(network=net,engine=eng,B=B,rows_per_second=y,prefix_rows=True) for B,y in pts]
f.set_xlim(.8,1600);f.set_ylim(10,2e4);f.set_xticks([1,16,256,1024]);f.set_xticklabels(['1','16','256','1,024']);f.set_yticks([1e1,1e2,1e3,1e4]);f.set_ylabel('Rows per second',fontsize=7.5);f.set_xlabel(r'Adjoint chunk size $B$',fontsize=7.5)
f.legend(handles=[Line2D([],[],color=GREY,ls=DOT,label='Gather'),Line2D([],[],color=GREY,ls=DASH,label='SpMM'),Line2D([],[],color=GREY,label='Automatic')],loc='lower right',fontsize=7,handlelength=1.6,labelspacing=.2,borderpad=.1)
ft(4,1.9,'Fixed hydraulics · supported first-order reactions · compatible recorded branches',7,color=GREY)

# Source-data exports keep exact saved values (no error floors or jitter).
out=pf.OUT
def csvout(name,rows):
 with (out/name).open('w',newline='',encoding='utf-8') as fp:wr=csv.DictWriter(fp,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
bc=[dict(network=r['network'],fused_terms=int(terms[i]),forward_rel=float(forward[i]),adjoint_rel=float(adjoint[i]),fd_rel=None if not np.isfinite(worst[i]) else float(worst[i]),fd_worst_direction=directions[i],reactflag_off=bool(off[i])) for i,r in enumerate(g2)]
dcsv=[dict(case=dd['case'],parameter=dd['param'],mode=mode,h=x['h'],relative_error=x['rel_err']) for dd in sweeps for mode in ['M1_ctol0','M0_inp_ctol'] for x in dd['runs'][mode]['sweep']]
for name,rows in [('fig1_source_bc.csv',bc),('fig1_source_d.csv',dcsv),('fig1_source_e.csv',ecsv),('fig1_source_f.csv',fcsv)]:csvout(name,rows)
manifest=dict(dimensions_mm=[W,H],source_sha256=sources,schematic='Illustrative; no measured data in panel a.',caveats=['c: circles use fused verification; FD triangles use separately saved Ctol=0 per-operator verification and show the worst checked parameter-group direction.','d: upper includes six reaction directions and one affine initial-quality direction; lower includes four zero-coefficient/branch-boundary directions.','e: CUDA-graph forward+backward divided by batch size; library FD is projected; setup excluded.','f: prefix-row throughput; all BWSN2 rows timed here occur at t=0 and have zero kw sensitivity; ordinary PIPE count excludes two CVPIPEs.'],g1_cases=len(g1),g1_segments=sum(r['segments_created'] for r in g1),g2_configs=len(g2),fd_configs=int(valid.sum()))
(out/'fig1_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for ext in ['svg','pdf','png']:fig.savefig(out/f'fig1_engine.{ext}',dpi=600)
pf._png_to_tif(out/'fig1_engine.png',out/'fig1_engine.tif')
fig.savefig(out/'fig1_engine_preview.png',dpi=220);plt.close(fig)
print(f'Figure 1: {W} x {H} mm; {len(g2)} configurations, {valid.sum()} FD checks, {len(sweeps)} sweeps.')
