"""Read-only artwork/source checks for the redesigned Figures 2 and 3."""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path
import numpy as np
import fitz
from PIL import Image
import paperfig as pf

def rows(name):
    with (pf.OUT/name).open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))
def num(value):return None if value in ('',None) else float(value)
def same(a,b):
    assert a==b,(a,b)
report={}
for stem in ['fig2_field','fig3_exp1']:
    m=json.loads((pf.OUT/f'{stem}_manifest.json').read_text())
    for name,sha in m['source_sha256'].items():
        root=pf.REPO if name.startswith(('networks/','EPANET-dev/')) else pf.RES
        same(hashlib.sha256((root/name).read_bytes()).hexdigest(),sha)
    with fitz.open(pf.OUT/f'{stem}.pdf') as pdf:
        page=pdf[0];spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        assert all(s['bbox'][0]>=0 and s['bbox'][1]>=0 and s['bbox'][2]<=page.rect.width and s['bbox'][3]<=page.rect.height for s in spans)
        assert min(s['size'] for s in spans)>=4.9
    with Image.open(pf.OUT/f'{stem}.tif') as im:same([float(v) for v in im.info['dpi']],[600.,600.])
    report[stem]={'source_hashes_verified':len(m['source_sha256']),'text_within_page':True,'minimum_glyph_pt':min(s['size'] for s in spans)}

cal={s:next(d for d in pf.load(pf.DATA['calib']/f'g4_fused_{s}_cuda.json')['datasets'] if not d['fix_kb']) for s in ['Fairfield','Bellingham','NorthPenn']}
for r in rows('fig2_source_calibration_runs.csv'):
    d=cal[r['system']]['runs'][int(r['run'])];same(r['method'],d['method'])
    for k in ['wall_s','n_eval','rmse']:same(num(r[k]),d[k])
    same(r['success'],'' if d.get('success') is None else str(d['success']))
h2={d['dataset']:d for d in pf.load(pf.DATA['h2']/'h2_uq.json')['datasets']}
for r in rows('fig2_source_metrics.csv'):
    if r['panel'] in ['a','b']:
        rr=[q for q in cal[r['system']]['runs'] if q['method']==r['method']]
        value=st.median(q['wall_s' if r['panel']=='a' else 'rmse'] for q in rr)
        same(float(r['value']),value)
        if r['panel']=='a':same(int(r['n_eval']),int(st.median(q['n_eval'] for q in rr)));assert .08<=value<=90
        else:assert .1<=value<=.285
    else:
        d=h2[r['system']]['loso'];key='pred_lin' if r['method']=='predictive' else 'param_only_lin'
        value=d['param_share_of_pred_var'] if r['metric']=='share predictive variance' else sum(q['n']*q[key] for q in d['site_coverage95'].values())/sum(q['n'] for q in d['site_coverage95'].values())
        same(float(r['value']),value)
        assert 0<=value<=1.04
field={d['dataset']:d for d in pf.load(pf.DATA['exp3']/'exp3_field.json')['field']}
for r in rows('fig2_source_pipe_ranks.csv'):
    d=field[r['system']];arr=d['pipe_sd_over_tau'] if r['stage']=='existing' else d['prescription']['greedy']['logdet']['pipe_sd_over_tau_after'];same(float(r['sd_over_tau']),sorted(arr)[int(r['rank'])-1]);assert 1<=int(r['rank'])<=600 and 0<=float(r['sd_over_tau'])<=1.05
for r in rows('fig2_source_maps.csv'):
    d=field[r['system']];idx=list(map(str,d['pipe_ids'])).index(r['pipe_id']);same(float(r['posterior_prior_sd']),d['prescription']['greedy']['logdet']['pipe_sd_over_tau_after'][idx])
    v=float(r['posterior_prior_sd']);same(r['classification'],'ident' if v<.5 else 'part' if v<.9 else 'prior')
geometry={s:pf.read_inp_geometry(pf.INP[s],pf.MAP.get(s))[0] for s in ['Net3','NorthPenn','Bellingham']}
for r in rows('fig2_source_sites.csv'):
    d=field[r['system']];names=d['sites'] if r['stage']=='existing' else d['prescription']['greedy']['logdet']['added_sites'];assert r['node_id'] in names
    pos=geometry[r['system']].get(r['node_id']);same(num(r['x']),None if pos is None else pos[0]);same(num(r['y']),None if pos is None else pos[1])
    if r['stage']=='added':same(int(r['added_order']),names.index(r['node_id'])+1)

data={s:pf.load(pf.DATA['exp1']/f'exp1_{s}.json')['networks'][0] for s in ['Net3','NorthPenn']}
for r in rows('fig3_source_maps.csv'):
    n=data[r['network']];node=r['node_id'];pos=geometry[r['network']].get(node)
    same(num(r['x']),None if pos is None else pos[0]);same(num(r['y']),None if pos is None else pos[1]);same(r['is_candidate'],str(node in n['candidates']))
    support=next(q for q in n['refit'] if abs(q['lam_frac']-.4)<1e-9)['stations'];same(r['in_lasso'],str(node in support));gg=[q['added'] for q in n['greedy']];same(num(r['greedy_order']),gg.index(node)+1 if node in gg else None)
for r in rows('fig3_source_designs.csv'):
    n=data[r['network']];d=n['plant_only'] if r['method']=='plant_only' else n[r['method']][int(r['design'])]
    same(float(r['coverage']),d['coverage'])
    if r['method']!='plant_only':same(float(r['mass_kg_d']),d['mass_kg_d']);same(int(r['n_stations']),d['n_stations'])
    if r['method'] in ['refit','const_rate']:same(float(r['lam_frac']),d['lam_frac'])
    if r['method']!='plant_only':assert .85<=float(r['coverage'])<=1.003 and float(r['mass_kg_d'])>0
for r in rows('fig3_source_dosing.csv'):
    n=data['Net3'];d=next(q for q in n['refit'] if abs(q['lam_frac']-.4)<1e-9);v=d['u'][n['candidates'].index(r['node_id'])][int(r['hour'])]
    same(float(r['raw_u']),v);same(float(r['rate_kg_h']),v*n['settings']['mass_coef_kg_d_per_unit'])
for r in rows('fig3_source_curves.csv'):
    n=data[r['network']];d=n['plant_only'] if r['method']=='plant_only' else next(q for q in n['refit'] if q['lam_frac']==float(r['method'].split('_')[1]));curve=d['curves'];idx=curve['t_h'].index(float(r['time_h']));same(float(r['fraction_nodes_below_lo']),curve['frac_below_lo'][idx])
for stem in ['fig2','fig3']:
    report[stem+'_source_rows']={p.name:len(rows(p.name)) for p in sorted(pf.OUT.glob(stem+'_source_*.csv'))}
report['visual_review']='Latest PNG previews inspected after adjusting map number offsets; no text clipping or obstructing legends observed.'
report['result']='Quantitative source values, map coordinates, input hashes, minimum glyph size, TIFF resolution and page bounds checked; no experiments rerun.'
(pf.OUT/'fig23_data_qa.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
