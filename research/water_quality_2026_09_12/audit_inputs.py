"""Read-only source inventory; generated audit files remain beside this script."""
from pathlib import Path
import hashlib
import json
from collections import Counter
import fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sections(path):
    result = {}
    section = ''
    for number, raw in enumerate(path.read_text(encoding='utf-8-sig', errors='replace').splitlines(), 1):
        line = raw.split(';')[0].strip()
        if line.startswith('[') and ']' in line:
            section = line.split(']')[0][1:].upper()
            result.setdefault(section, [])
        elif line:
            result.setdefault(section, []).append({'line': number, 'text': line})
    return result

def main():
    records = []
    for path in sorted(ROOT.rglob('*.inp')):
        if 'research' in path.relative_to(ROOT).parts:
            continue
        sec = sections(path)
        options = sec.get('OPTIONS', [])
        qual = next((r['text'] for r in options if r['text'].split()[0].upper() == 'QUALITY'), '(default NONE)')
        record = {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
                  'counts': {k: len(sec.get(k, [])) for k in ['JUNCTIONS','RESERVOIRS','TANKS','PIPES','PUMPS','VALVES','QUALITY','SOURCES','REACTIONS','MIXING']},
                  'quality_mode': qual,
                  'options': options, 'times': sec.get('TIMES', []),
                  'reactions': sec.get('REACTIONS', []), 'sources': sec.get('SOURCES', [])}
        records.append(record)
    msx = []
    for path in sorted((ROOT / 'networks').rglob('*.msx')):
        sec = sections(path)
        msx.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
                    'declared_species': len(sec.get('SPECIES',[])),
                    'pipe_equations': dict(Counter(x['text'].split()[0].upper() for x in sec.get('PIPE',[]))),
                    'tank_equations': dict(Counter(x['text'].split()[0].upper() for x in sec.get('TANK',[]))),
                    'options': sec.get('OPTIONS', [])})
    pdfs = []
    text_dir = OUT / 'pdf_text'
    text_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(ROOT.rglob('*')):
        if path.suffix.lower() != '.pdf' or 'research' in path.relative_to(ROOT).parts:
            continue
        doc = fitz.open(path)
        short_hash = digest(path)[:12]
        texts = [page.get_text() for page in doc]
        destination = text_dir / (path.stem + '_' + short_hash + '.txt')
        destination.write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n{text}' for i,text in enumerate(texts)), encoding='utf-8')
        pdfs.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
                     'pages':len(doc), 'text_chars':sum(map(len,texts)),
                     'extracted_text': destination.relative_to(OUT).as_posix(),
                     'first_page_excerpt': texts[0][:1400] if texts else ''})
        doc.close()
    result = {'inp': records, 'msx': msx, 'pdf': pdfs}
    (OUT/'input_inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    network_records=[r for r in records if r['path'].startswith('networks/')]
    summary={'inp_all':len(records),'inp_networks':len(network_records),
             'inp_networks_unique_bytes':len({r['sha256'] for r in network_records}),
             'quality_modes_networks':dict(Counter(r['quality_mode'] for r in network_records)),
             'msx_files':len(msx),'msx_unique_bytes':len({r['sha256'] for r in msx}),
             'pdf_files':len(pdfs),'pdf_pages':sum(r['pages'] for r in pdfs),
             'pdf_unique_bytes':len({r['sha256'] for r in pdfs})}
    (OUT/'inventory_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for r in pdfs:
        print(r['path'],r['pages'],r['text_chars'])

if __name__ == '__main__':
    main()
