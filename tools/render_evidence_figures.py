"""Render simple SVGs directly from recovered logs; no model execution."""
from pathlib import Path
import re,json,csv
ROOT=Path(__file__).resolve().parents[1] if 'tools' in str(Path(__file__).resolve()) else Path('minimind-llm-training')
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
pat=re.compile(r'Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), (?:loss: ([\d.]+)|Reward: (-?[\d.]+))')
summary={}
for name in ['pretrain_10k','pretrain_full','sft_full','grpo_full_resume']:
 data=(ROOT/f'evidence/training/{name}.log').read_bytes().decode('utf-8');rows=[]
 for line_no,line in enumerate(data.split('\n'),1):
  m=pat.search(line)
  if m:rows.append((line_no,int(m[3]),int(m[4]),float(m[5] or m[6])))
 summary[name]={'logged_points':len(rows),'first':rows[0],'last':rows[-1],'first_10_mean':sum(x[3] for x in rows[:10])/min(10,len(rows)),'last_10_mean':sum(x[3] for x in rows[-10:])/min(10,len(rows))}
 with (ROOT/f'evidence/training/{name}_parsed.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['original_line','batch_step','total_batches','training_loss' if name!='grpo_full_resume' else 'reward']);writer.writerows(rows)
 if name=='pretrain_10k':continue
 w,h=840,330;lo=min(x[3] for x in rows);hi=max(x[3] for x in rows);xmax=rows[-1][2]
 def xy(x):return f'{65+x[1]/xmax*740:.2f},{265-(x[3]-lo)/(hi-lo or 1)*205:.2f}'
 pts=' '.join(map(xy,rows));metric='Training cross-entropy loss' if name!='grpo_full_resume' else 'Logged group reward (not held-out quality)'
 svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="840" height="330" viewBox="0 0 840 330"><rect width="840" height="330" fill="white"/><g font-family="Arial,sans-serif" fill="#1e293b"><text x="65" y="27" font-size="18">{name}: {metric}</text><text x="65" y="48" font-size="12">Recovered training log; unsmoothed logged points; no model rerun</text><path d="M65 60V265H805" fill="none" stroke="#64748b"/><polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="1"/><text x="8" y="67" font-size="12">{hi:.2f}</text><text x="8" y="267" font-size="12">{lo:.2f}</text><text x="65" y="285" font-size="12">0</text><text x="745" y="285" font-size="12">{xmax}</text><text x="340" y="311" font-size="13">Dataloader batch step</text></g></svg>'''
 (OUT/f'{name}.svg').write_text(svg)
(ROOT/'evidence/TRAINING_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
