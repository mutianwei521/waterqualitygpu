"""Figure 3: booster support, dosing trade-offs and temporal concentration shortfall.

The complete saved refit/constant/greedy designs are retained; support maps and the
Net3 dosing heatmap show lambda=0.4 lambda0. A group-lasso path is a continuous
relaxation, not a certificate of combinatorial optimality. Not all refits satisfy
their optimizer stopping criterion. No simulation or fitting is rerun here.
"""
import re
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import paperfig as pf
import fig23_style as fs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
fs.setup()
NETS=[('Net3','Net3',fs.MAGENTA),('NorthPenn','North Penn',fs.TEAL)]
data={k:fs.load(pf.DATA['exp1']/f'exp1_{k}.json')['networks'][0] for k,_,_ in NETS}
def sources_from_inp(path):
    ids=[];section=None
    with open(path,encoding='latin-1') as handle:
        for raw in handle:
            line=raw.split(';',1)[0].strip()
            if not line:continue
            m=re.match(r'\[(\w+)\]',line)
            if m:section=m.group(1).upper();continue
            if section in ('SOURCES','RESERVOIRS'):ids.append(line.split()[0])
    return ids
glyph={'greedy':dict(marker='o',ms=3,mew=.5),'refit':dict(marker='s',mfc='white',ms=3.2,mew=.8),'const_rate':dict(marker='^',mfc='white',ms=3.5,mew=.8)}
fig=plt.figure(figsize=(176/25.4,190/25.4));maps=[];summary={};designs=[]
for j,(key,label,col) in enumerate(NETS):
    n=data[key];x0=6+j*58;ax=fs.axis(fig,x0,118,49,55)
    fs.panel(fig,x0-2,184,'ab'[j],label);fs.text(fig,x0,178,f'{n["sizes"]["nodes"]} nodes · {n["sizes"]["candidates"]} candidates',7.1,color=fs.GREY)
    fs.record_file(pf.INP[key])
    if pf.MAP.get(key):fs.record_file(pf.MAP[key])
    coords,links=pf.read_inp_geometry(pf.INP[key],pf.MAP.get(key));pf.draw_network(ax,coords,links,color=fs.LIGHT,lw=.5,pad=.08)
    cand=[s for s in n['candidates'] if s in coords];src=[s for s in sources_from_inp(pf.INP[key]) if s in coords]
    ax.scatter([coords[s][0] for s in cand],[coords[s][1] for s in cand],s=1.6,color=fs.MID,lw=0,zorder=2)
    ax.scatter([coords[s][0] for s in src],[coords[s][1] for s in src],s=43,marker='*',fc=fs.GOLD,ec=fs.INK,lw=.5,zorder=5)
    ref=next(r for r in n['refit'] if abs(r['lam_frac']-.4)<1e-9);lasso=[s for s in ref['stations'] if s in coords]
    ax.scatter([coords[s][0] for s in lasso],[coords[s][1] for s in lasso],s=22,marker='s',fc='none',ec=col,lw=.9,zorder=4)
    greedy=[g['added'] for g in n['greedy']]
    for order,s in enumerate(greedy,1):
        if s in coords:
            ax.scatter(*coords[s],s=26,marker='^',fc=col,ec=fs.INK,lw=.5,zorder=6)
            offset={('Net3',4):(-10,-3),('Net3',6):(-3,-10),('Net3',7):(4,7),('Net3',8):(7,-1),('NorthPenn',1):(-8,7),('NorthPenn',2):(-9,-6),('NorthPenn',4):(10,2),('NorthPenn',7):(10,-10)}.get((key,order),(3,4))
            ax.annotate(str(order),coords[s],offset,textcoords='offset points',fontsize=7.1,color=fs.INK,fontweight='bold',zorder=7,
                        arrowprops=dict(arrowstyle='-',color=fs.GREY,lw=.4,shrinkA=1,shrinkB=3) if max(abs(t) for t in offset)>=8 else None)
    fs.text(fig,x0,114,f'Group lasso: {len(ref["stations"])} stations',7.2,color=col)
    fs.text(fig,x0,110,'Greedy: all 8 selected stations',7.1,color=fs.GREY)
    summary[key]=dict(greedy=greedy,lasso=ref['stations'],missing=[s for s in greedy+ref['stations'] if s not in coords],plant=n['plant_only']['coverage'])
    for s in dict.fromkeys(n['candidates']+src):
        maps.append(dict(network=key,node_id=s,is_candidate=s in n['candidates'],is_existing_source=s in src,in_lasso=s in ref['stations'],greedy_order=greedy.index(s)+1 if s in greedy else None,x=coords[s][0] if s in coords else None,y=coords[s][1] if s in coords else None))
    for method in ['refit','const_rate','greedy']:
        for i,r in enumerate(n[method]):designs.append(dict(network=key,method=method,design=i,lam_frac=r.get('lam_frac'),n_stations=r['n_stations'],coverage=r['coverage'],mass_kg_d=r['mass_kg_d'],converged=r.get('converged')))
    designs.append(dict(network=key,method='plant_only',design=0,n_stations=0,coverage=n['plant_only']['coverage'],mass_kg_d=0))
fs.key(fig,6,103,'Candidate',fs.MID,'o',ls='none',filled=True);fs.key(fig,51,103,'Existing chlorine source',fs.GOLD,'*',ls='none',filled=True)
fs.key(fig,6,98,r'Lasso support, $\lambda/\lambda_0=0.4$',fs.GREY,'s',ls='none',size=8.0);fs.key(fig,65,98,'Greedy selection order',fs.GREY,'^',ls='none',filled=True)

c=fs.axis(fig,130,125,42,48);fs.style(c,fs.PALE_TEAL);fs.panel(fig,120,184,'c','Coverage and stations')
fs.text(fig,130,178,r'Threshold: $c\geq0.2$ mg/L',7.2,color=fs.GREY)
for key,label,col in NETS:
    n=data[key];xg=[0]+[g['n_stations'] for g in n['greedy']];yg=[n['plant_only']['coverage']]+[g['coverage'] for g in n['greedy']]
    c.plot(xg,yg,color=col,lw=1.05,**glyph['greedy'])
    for method in ['refit','const_rate']:
        rr=sorted(n[method],key=lambda r:r['n_stations']);c.plot([r['n_stations'] for r in rr],[r['coverage'] for r in rr],ls='none',color=col,**glyph[method])
c.set_xscale('symlog',linthresh=8,linscale=1.2);c.set_xlim(-.3,130);c.set_xticks([0,4,8,20,50,100]);c.set_xticklabels(['0','4','8','20','50','100']);c.set_ylim(.80,1.003);c.set_yticks([.8,.9,1.]);c.set_xlabel('Booster stations');c.set_ylabel('Node-hour coverage (fraction)')
for y,lab,mk in [(113,'Greedy path','o'),(108,'Lasso support, refit','s'),(103,'Same support, constant','^')]:fs.key(fig,130,y,lab,fs.GREY,mk,ls='-' if mk=='o' else 'none',filled=mk=='o')

d=fs.axis(fig,15,27,43,51);fs.style(d);fs.panel(fig,4,88,'d','Dose–coverage trade-off')
for key,label,col in NETS:
    n=data[key]
    for aa,bb in zip(n['refit'],n['const_rate']):
        assert aa['lam_frac']==bb['lam_frac']
        d.plot([aa['mass_kg_d'],bb['mass_kg_d']],[aa['coverage'],bb['coverage']],color=col,lw=.55,alpha=.4)
    for method in ['refit','const_rate','greedy']:d.plot([r['mass_kg_d'] for r in n[method]],[r['coverage'] for r in n[method]],ls='none',color=col,**glyph[method])
mass=[r['mass_kg_d'] for r in designs if r['method']!='plant_only'];assert min(mass)>0
d.set_xscale('log');pf.log_ticks(d,'x');d.tick_params(axis='x',labelsize=8.0);d.set_xlim(min(mass)/1.5,max(mass)*1.3);d.set_ylim(.85,1.003);d.set_yticks([.85,.9,.95,1]);d.set_xlabel('Chlorine mass (kg/d)');d.set_ylabel('Node-hour coverage (fraction)')
fs.text(fig,15,83,'Links pair the same support',7.1,color=fs.GREY)
fs.key(fig,15,15,'Net3',fs.MAGENTA);fs.key(fig,38,15,'North Penn',fs.TEAL)

e=fs.axis(fig,79,28,38,48);fs.panel(fig,69,88,'e','Hourly dosing')
n=data['Net3'];ref=next(r for r in n['refit'] if abs(r['lam_frac']-.4)<1e-9);U=np.array(ref['u']);rows=np.flatnonzero(np.abs(U).sum(axis=1)>0);coef=n['settings']['mass_coef_kg_d_per_unit'];Z=U[rows]*coef
# For Net3's 24 one-hour slots, this coefficient converts mg/min to kg/h.
assert U.shape[1]==24 and np.isclose(coef,6e-5)
# A monotone rose ramp keeps panel e visually coherent while still resolving the
# four-order-of-magnitude positive range.  Exact zeros occupy a narrow neutral
# plateau; the smallest nonzero value crosses immediately into pale pink after the
# square-root transform.  Increasingly dark berry tones therefore encode magnitude
# without the false categorical boundaries created by the former multihue ramp.
rose_anchors=[
    (0.000,'#F7F5F6'), (0.002,'#F7F5F6'), (0.003,'#FAD9E8'),
    (0.080,'#F7C3DA'), (0.200,'#F0A1C2'), (0.380,'#E477A8'),
    (0.580,'#D14A8A'), (0.760,'#B32970'), (0.900,'#861951'),
    (1.000,'#50102F')]
cmap=LinearSegmentedColormap.from_list('dose_rose',rose_anchors,N=1024)
dose_norm=PowerNorm(gamma=0.5,vmin=0,vmax=Z.max())
im=e.imshow(Z,aspect='auto',cmap=cmap,norm=dose_norm,interpolation='nearest',origin='upper');e.grid(False)
e.set_yticks(np.arange(len(rows)));e.set_yticklabels([n['candidates'][i] for i in rows]);e.set_xticks([0,6,12,18,23]);e.set_xticklabels(['0','6','12','18','23']);e.set_xlabel('Hour of day');e.set_ylabel('Station (node ID)');e.tick_params(length=0,pad=2)
# Fine white cell rules improve local comparison and remain unobtrusive after reduction.
e.set_xticks(np.arange(-.5,Z.shape[1],1),minor=True);e.set_yticks(np.arange(-.5,Z.shape[0],1),minor=True)
e.grid(which='minor',color='white',lw=.18,alpha=.72);e.tick_params(which='minor',bottom=False,left=False)
for sp in e.spines.values():sp.set_visible(False)
fs.text(fig,79,82,f'Net3 · {len(rows)} stations · {ref["mass_kg_d"]:.2f} kg/d',7.1,color=fs.MAGENTA)
cax=fs.axis(fig,79,16,38,2.1);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.001,.005,.02,.05]);cb.ax.set_xticklabels(['0','.001','.005','.02','.05']);cb.ax.tick_params(labelsize=6.7,length=1.5,pad=1);cb.outline.set_visible(True);cb.outline.set_edgecolor('#6E3652');cb.outline.set_linewidth(.4)
fs.text(fig,98,11.0,'Square-root colour scale',6.5,color=fs.GREY,ha='center')
fs.text(fig,98,7.0,'Injection rate (kg/h)',7.2,ha='center')
dose=[dict(network='Net3',node_id=n['candidates'][i],hour=h,rate_kg_h=float(U[i,h]*coef),raw_u=float(U[i,h])) for i in rows for h in range(24)]
mass_check=float(Z.sum());assert abs(mass_check-ref['mass_kg_d'])<1e-7

fs.panel(fig,124,88,'f','Temporal node shortfall');curves=[]
for j,(key,label,col) in enumerate(NETS):
    ax=fs.axis(fig,135,58-j*31,37,18);fs.style(ax);n=data[key];p=n['plant_only']['curves'];picks=[next(r for r in n['refit'] if abs(r['lam_frac']-v)<1e-9) for v in [.4,.1]]
    ax.plot(p['t_h'],p['frac_below_lo'],color=fs.GREY,lw=1.0)
    for rr,ls in zip(picks,['-',fs.DASH]):ax.plot(rr['curves']['t_h'],rr['curves']['frac_below_lo'],color=col,ls=ls,lw=1.05)
    for meth,rr in [('plant_only',p)]+[(f'lambda_{r["lam_frac"]}',r['curves']) for r in picks]:curves += [dict(network=key,method=meth,time_h=t,fraction_nodes_below_lo=y) for t,y in zip(rr['t_h'],rr['frac_below_lo'])]
    ymax=max(max(p['frac_below_lo']),*(max(r['curves']['frac_below_lo']) for r in picks));ax.set_ylim(0,ymax*1.15)
    fs.text(fig,135,81-j*31,f'{label}: {picks[0]["n_stations"]} / {picks[1]["n_stations"]} stations',7.1,color=col)
    if j==1:ax.set_xlabel('Time (h)')
fs.text(fig,125.3,50,'Nodes below 0.2 mg/L (fraction)',7.2,rotation=90,ha='center')
fs.key(fig,135,15,'Plant only',fs.GREY);fs.key(fig,135,10,r'$\lambda/\lambda_0=0.4$',fs.INK,size=8.0);fs.key(fig,135,5,r'$\lambda/\lambda_0=0.1$',fs.INK,ls=fs.DASH,size=8.0)
for name,rr in [('designs',designs),('maps',maps),('dosing',dose),('curves',curves)]:fs.source_csv(f'fig3_source_{name}.csv',rr)
fs.finish(fig,'fig3_exp1',__doc__,dict(networks=summary,data_rows={'designs':len(designs),'map_candidates':len(maps),'dosing':len(dose),'curves':len(curves)},heatmap_mass_kg_d=mass_check,reported_heatmap_mass_kg_d=ref['mass_kg_d'],heatmap_colour_normalization={'type':'PowerNorm','gamma':0.5,'colourmap':'dose_rose','vmin_kg_h':0.0,'vmax_kg_h':float(Z.max()),'colourbar_ticks_kg_h':[0,.001,.005,.02,.05],'zero_colour':'#F7F5F6','minimum_positive_colour':'#FAD9E8','tick_values_untransformed':True},caveats=['The symlog station axis retains the zero-station baseline and a linear segment through eight stations.','Every saved refit, constant-rate design and greedy stage is shown; no optimum certificate is inferred from visual paths.','Temporal shortfall is the fraction of nodes; aggregate coverage uses the experiment node-hour weights.','The dose heatmap uses every nonzero station in the selected design and all 24 slots.','Panel e uses a square-root (PowerNorm gamma=0.5) rose colour mapping to reveal low nonzero rates; colour-bar ticks report untransformed kg/h values and exact zeros retain the neutral endpoint.']))
