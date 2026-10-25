"""Figure 2: efficient field calibration, held-out prediction and limited pipe information.

Mixed quantitative/map figure. Medians, coverage weights, every pipe-rank value and
original network coordinates are preserved. Sensor prescriptions are local linear
predictions, not observations collected after installing new sensors.
"""
import statistics as st
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import paperfig as pf
import fig23_style as fs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
fs.setup()
SYS=['Fairfield','Bellingham','NorthPenn']
LABEL={'Fairfield':'Fairfield','Bellingham':'Bellingham','NorthPenn':'North Penn'}
NET={'Fairfield':fs.MAGENTA,'Bellingham':fs.CYAN,'NorthPenn':fs.TEAL}
METHODS=[('FD-LBFGSB','FD, zero tolerance',fs.GOLD,'o'),('FD-LBFGSB@ctol0.01','FD, default tolerance',fs.VERM,'^'),('ADJ-LBFGSB','Adjoint, eager GPU',fs.CYAN,'s'),('ADJ-LBFGSB-graph','Adjoint, CUDA graph',fs.PURPLE,'D')]
calib={s:next(d for d in fs.load(pf.DATA['calib']/f'g4_fused_{s}_cuda.json')['datasets'] if not d['fix_kb']) for s in SYS}
by={s:{} for s in SYS};raw=[]
for s in SYS:
    for i,r in enumerate(calib[s]['runs']):
        by[s].setdefault(r['method'],[]).append(r)
        raw.append(dict(system=s,run=i,method=r['method'],wall_s=r['wall_s'],n_eval=r['n_eval'],rmse=r['rmse'],success=r.get('success')))
h2={d['dataset']:d for d in fs.load(pf.DATA['h2']/'h2_uq.json')['datasets']}
field={d['dataset']:d for d in fs.load(pf.DATA['exp3']/'exp3_field.json')['field']}
def wcov(d,key):
    sc=d['loso']['site_coverage95'];return sum(v['n']*v[key] for v in sc.values())/sum(v['n'] for v in sc.values())
fig=plt.figure(figsize=(176/25.4,184/25.4))
a=fs.axis(fig,15,112,43,43);b=fs.axis(fig,74,112,43,43);c=fs.axis(fig,134,112,38,43)
for ax in [a,b,c]:fs.style(ax)
for xx,let,title in [(4,'a','Calibration time'),(64,'b','Fit to field data'),(124,'c','Held-out coverage')]:fs.panel(fig,xx,179,let,title)
x=np.arange(3);w=.18;metric=[]
for k,(m,lab,col,mk) in enumerate(METHODS):
    fs.key(fig,15,172-k*4.1,lab,col,mk,ls='none',filled=True)
    for xi,s in zip(x+(k-1.5)*w,SYS):
        v=st.median(r['wall_s'] for r in by[s][m]);n=int(st.median(r['n_eval'] for r in by[s][m]))
        a.plot([xi,xi],[.08,v],color=col,lw=2.8,alpha=.3,solid_capstyle='butt');a.scatter(xi,v,s=16,marker=mk,fc=col,ec=col,lw=.5,zorder=3)
        a.text(xi,v*1.24,str(n),ha='center',va='bottom',rotation=90,fontsize=7.1,color=fs.GREY)
        metric.append(dict(panel='a',system=s,method=m,value=v,metric='median wall_s',n_runs=len(by[s][m]),n_eval=n))
a.set_yscale('log');a.set_ylim(.08,90);a.set_yticks([.1,1,10]);pf.log_ticks(a,'y');a.set_ylabel('Time per calibration (s)')
fs.text(fig,15,95,'Numbers: loss evaluations per fit',7.1,color=fs.GREY)
spec=[('nominal(INP)','Input parameters','o','white',fs.GREY,-.24),('FD-LBFGSB','FD, zero tolerance','x',fs.GOLD,fs.GOLD,-.12),('ADJ-LBFGSB-graph','Adjoint, CUDA graph','D',fs.PURPLE,fs.PURPLE,0),('FD-LBFGSB@ctol0.01','FD, default tolerance','^',fs.VERM,fs.VERM,.12),('FD-LBFGSB@ctol0.01,eps1e-3','Default FD, h = 0.001','v',fs.MAGENTA,fs.MAGENTA,.24)]
vals={m:[st.median(r['rmse'] for r in by[s][m]) for s in SYS] for m,*_ in spec}
for j,(_,lab,mk,fc,ec,_) in enumerate(spec):fs.key(fig,74,174-j*4.05,lab,ec,mk,ls='none',filled=fc!='white')
for xi in x:b.fill_between([xi-.34,xi+.34],min(vals[m][xi] for m,*_ in spec[1:]),vals['nominal(INP)'][xi],color=fs.PALE_PURPLE,zorder=.5)
for m,lab,mk,fc,ec,dx in spec:
    kw=dict(color=ec) if mk=='x' else dict(facecolor=fc,edgecolor=ec)
    b.scatter(x+dx,vals[m],marker=mk,s=18,lw=.8,zorder=3,**kw)
    metric += [dict(panel='b',system=s,method=m,value=v,metric='median rmse',n_runs=len(by[s][m])) for s,v in zip(SYS,vals[m])]
b.set_ylim(.10,.285);b.set_yticks([.1,.15,.2,.25]);b.set_ylabel('Chlorine RMSE (mg/L)')
fs.text(fig,74,95,'Same observations; different solvers',7.1,color=fs.GREY)
fs.key(fig,134,172,'Parameters + noise',fs.TEAL,'o',ls='none',filled=True);fs.key(fig,134,167.9,'Parameters only',fs.PURPLE,'s',ls='none')
for xi,s in zip(x,SYS):
    p=wcov(h2[s],'pred_lin');q=wcov(h2[s],'param_only_lin');share=h2[s]['loso']['param_share_of_pred_var']
    c.plot([xi,xi],[q,p],color=fs.LIGHT,lw=2);c.scatter(xi,p,s=23,fc=fs.TEAL,ec=fs.TEAL,zorder=3);c.scatter(xi,q,s=20,marker='s',fc='white',ec=fs.PURPLE,lw=.8,zorder=3)
    c.text(xi,q-.055,f'{share:.1%}',fontsize=7.1,color=fs.PURPLE,ha='center',va='top')
    metric += [dict(panel='c',system=s,method=method,metric=met,value=v) for method,met,v in [('predictive','coverage95',p),('parameter-only','coverage95',q),('parameter-only','share predictive variance',share)]]
c.axhline(.95,color=fs.GREY,ls=fs.DASH,lw=.7);c.set_ylim(0,1.04);c.set_yticks([0,.25,.5,.75,1]);c.set_ylabel('Coverage of 95% interval')
fs.text(fig,134,160,'Reference: 95% coverage',7.1,color=fs.GREY);fs.text(fig,134,95,'Numbers: parameter share',7.1,color=fs.GREY);fs.text(fig,134,91.2,'of predictive variance',7.1,color=fs.GREY)
for ax in [a,b,c]:ax.set_xticks(x);ax.set_xticklabels([LABEL[s] for s in SYS],rotation=25,ha='right');ax.set_xlim(-.5,2.5)
d=fs.axis(fig,15,27,43,51);fs.style(d,fs.PALE_PURPLE);fs.panel(fig,4,87,'d','Pipe-level information')
rankrows=[];before_after={}
for j,s in enumerate(SYS):
    f=field[s];pre=np.sort(f['pipe_sd_over_tau']);post=np.sort(f['prescription']['greedy']['logdet']['pipe_sd_over_tau_after'])
    for arr,stage,ls in [(pre,'existing','-'),(post,'after D-optimal',fs.DASH)]:
        d.plot(np.arange(1,len(arr)+1),arr,color=NET[s],ls=ls,lw=1)
        rankrows += [dict(system=s,stage=stage,rank=i+1,sd_over_tau=float(v)) for i,v in enumerate(arr)]
    n0=f['audit']['n_identifiable'];n1=f['prescription']['greedy']['logdet']['n_identifiable_after'];before_after[s]=(n0,n1)
    d.text(.96,.34-j*.09,f'{LABEL[s]} {n0} → {n1}',transform=d.transAxes,ha='right',color=NET[s],fontsize=7.2)
d.axhline(.5,color=fs.GREY,ls=fs.DASH,lw=.6);d.text(.96,.51,'ratio < 0.5',transform=d.transAxes,ha='right',va='bottom',fontsize=7.1,color=fs.GREY)
d.set_xscale('log');pf.log_ticks(d,'x');d.set_xlim(1,600);d.set_ylim(0,1.05);d.set_yticks([0,.5,1]);d.set_xlabel('Pipe rank');d.set_ylabel('Posterior / prior sd of pipe $K_w$')
fs.key(fig,15,15,'Existing sites',fs.GREY);fs.key(fig,15,10,'+5 D-optimal sites',fs.GREY,ls=fs.DASH)
maprows=[];siterows=[];missing={}
for x0,s,letter in [(69,'NorthPenn','e'),(123,'Bellingham','f')]:
    ax=fs.axis(fig,x0,28,49,50);f=field[s];fs.panel(fig,x0-5,87,letter,LABEL[s]);fs.text(fig,x0,82,f'{f["n_pipes"]} pipes · {len(f["sites"])} existing sites',7.1,color=fs.GREY)
    fs.record_file(pf.INP[s])
    if pf.MAP.get(s):fs.record_file(pf.MAP[s])
    coords,links=pf.read_inp_geometry(pf.INP[s],pf.MAP.get(s));pf.draw_network(ax,coords,links,color='#E2E7EB',lw=.45,pad=.065)
    ends={lid:(n1,n2) for lid,n1,n2 in links};segs={'prior':[],'part':[],'ident':[]};miss=0
    for pid,v in zip(f['pipe_ids'],f['prescription']['greedy']['logdet']['pipe_sd_over_tau_after']):
        cls='ident' if v<.5 else 'part' if v<.9 else 'prior';ep=ends.get(str(pid).strip());matched=bool(ep and ep[0] in coords and ep[1] in coords)
        if matched:segs[cls].append([coords[ep[0]],coords[ep[1]]])
        else:miss+=1
        maprows.append(dict(system=s,pipe_id=pid,posterior_prior_sd=v,classification=cls,geometry_matched=matched))
    for cls,color,lw,z in [('prior',fs.LIGHT,.55,2),('part',fs.CYAN,1.15,3),('ident',fs.PURPLE,2.,4)]:ax.add_collection(LineCollection(segs[cls],colors=color,linewidths=lw,zorder=z,capstyle='round'))
    sites=[n for n in f['sites'] if n in coords];ax.scatter([coords[n][0] for n in sites],[coords[n][1] for n in sites],s=13,marker='s',fc='white',ec=fs.INK,lw=.6,zorder=5)
    added=f['prescription']['greedy']['logdet']['added_sites']
    for stage,names in [('existing',f['sites']),('added',added)]:
        siterows += [dict(system=s,node_id=node,stage=stage,added_order=names.index(node)+1 if stage=='added' else None,x=coords[node][0] if node in coords else None,y=coords[node][1] if node in coords else None) for node in names]
    for k,n in enumerate(added,1):
        if n in coords:ax.scatter(*coords[n],s=25,marker='^',fc=fs.GOLD,ec=fs.INK,lw=.5,zorder=6);ax.annotate(str(k),coords[n],(3,3),textcoords='offset points',fontsize=7.1,fontweight='bold',color=fs.INK,zorder=7)
    missing[s]=(miss,len(f['sites'])-len(sites),sum(n not in coords for n in added))
for y,label,color in [(22,'Prior-dominated: ≥0.9',fs.LIGHT),(17,'Partly informed: 0.5–0.9',fs.CYAN),(12,'Identifiable: <0.5',fs.PURPLE)]:fs.key(fig,69,y,label,color)
fs.key(fig,128,22,'Existing site',fs.INK,'s',ls='none');fs.key(fig,128,17,'Added site (order)',fs.GOLD,'^',ls='none',filled=True)
fs.text(fig,69,6,'Maps: local linear prescription at the fitted parameters',7.1,color=fs.GREY)
for name,rows in [('calibration_runs',raw),('metrics',metric),('pipe_ranks',rankrows),('maps',maprows),('sites',siterows)]:fs.source_csv(f'fig2_source_{name}.csv',rows)
fs.finish(fig,'fig2_field',__doc__,dict(geometry_missing=missing,identifiable_counts=before_after,data_rows={'runs':len(raw),'metrics':len(metric),'ranks':len(rankrows),'map_pipes':len(maprows)},caveats=['Timing and RMSE use original medians; source CSV retains all solver success flags.','LOSO coverage is weighted by observation count; labels are parameter shares of predictive variance.','+5-site covariance changes are local predictions, not a post-installation field trial.','Rank curves are independently sorted and do not pair the same pipe before and after.']))
