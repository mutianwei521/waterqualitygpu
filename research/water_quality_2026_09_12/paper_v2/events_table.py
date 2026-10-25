"""R2-5: table of the hydraulic events in the validation and application trajectories (final_5090_v2/r1_B2_*/hyd_events.json,
diffwq/hyd_events.py). Writes paper_v2/tables/T_S_events.md."""
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
f = sorted(glob.glob(str(ROOT / 'final_5090_v2' / 'r1_B2_*' / 'hyd_events.json')))[-1]
d = json.load(open(f, encoding='utf-8'))
head = ['case', 'nodes / links', 'pumps / valves / tanks', 'duration h', 'hydraulic periods', 'flow reversals (links)',
        'status changes: pumps / valves / pipes', 'tank fill-drain switches', 'junction-periods with >= 2 inflows', 'stagnant link-periods']
rows = [[c['case'], f"{c['nodes']:,} / {c['links']:,}", f"{c['pumps']} / {c['valves']} / {c['tanks']}", f"{c['duration_h']:.0f}", c['hydraulic_periods'],
         f"{c['flow_reversals']:,} ({c['links_reversing']:,})", f"{c['status_changes_pumps']} / {c['status_changes_valves']} / {c['status_changes_pipes']}",
         c['tank_fill_drain_switches'], f"{100 * c['junction_mixing_share']:.0f}% (up to {c['max_inflows_at_a_junction']})", f"{100 * c['stagnant_link_share']:.1f}%"]
        for c in d['cases']]
md = ['# Hydraulic events in the recorded trajectories (R2-5)\n',
      f"EPANET {d['engine_version']}; counted between consecutive hydraulic periods; stagnant threshold 0.005 gpm. Pipe status changes "
      'are check-valve closures and controls on pipes.\n',
      '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
(HERE / 'tables' / 'T_S_events.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print('\n'.join(md))
