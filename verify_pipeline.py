import sys
sys.stdout.reconfigure(encoding='utf-8')

from pipeline_utils import setup_logger, TR_SECONDS, REGION_ORDER, build_roi_indices

log = setup_logger('logs/pipeline_test.log')
log.info('TR_SECONDS=%.1f', TR_SECONDS)
log.info('REGION_ORDER=%s', REGION_ORDER)

indices = build_roi_indices()
print('\nROI Vertex Counts (Destrieux atlas on fsaverage5):')
for roi, idx in indices.items():
    print(f'  {roi:14s}: {len(idx):5d} vertices')

empty = [r for r, i in indices.items() if len(i) == 0]
if empty:
    print('FAIL: Empty ROIs:', empty)
    sys.exit(1)
else:
    print('\nPASS: All ROIs have vertices.')

# Test metadata loader
import csv
rows = []
with open('uploads/metadata.csv', newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        views_raw = r.get('views', '').strip()
        views = None
        if views_raw and views_raw.lower() not in ('null', 'none', ''):
            try:
                views = float(views_raw)
            except ValueError:
                pass
        rows.append({'id': int(r['id']), 'platform': r['platform'], 'label': r['label'], 'views': views})

print(f'\nMetadata rows: {len(rows)}')
print(f'  First row: {rows[0]}')
print(f'  Last row:  {rows[-1]}')
views_available = [r for r in rows if r['views'] is not None]
print(f'  Rows with views data: {len(views_available)}')
print('\nAll checks passed.')
