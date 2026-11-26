"""Verify output frame counts and record diagnostic ranges and test results."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
OUT=Path(__file__).resolve().parent/'results'

def run():
    r=json.loads((OUT/'report.json').read_text());d=np.load(OUT/'fields.npz')
    gif={}
    for f in sorted(OUT.glob('*.gif')):
        with Image.open(f) as im:gif[f.name]=im.n_frames
    assert len(gif)==9 and all(n==len(d['steps']) for n in gif.values()),gif
    r['gif_frame_counts']=gif
    r['field_snapshot_ranges']={k:[float(d[k].min()),float(d[k].max())] for k in ('phi','speed','jz','pressure','rho','mu','dynamic_pressure')}
    r['post_contact_peak_wall_pressure']=float(d['wall_pressure'][r['first_wall_wetting_step']:].max()) if r['first_wall_wetting_step'] is not None else None
    for key,f in [('new_tests','test_report.txt'),('full_regression','full_regression.txt')]:
        if (OUT/f).is_file():r[key]=(OUT/f).read_text().strip().splitlines()[-1]
    (OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Verified',len(gif),'GIFs,',len(d['steps']),'frames each')
if __name__=='__main__':run()
