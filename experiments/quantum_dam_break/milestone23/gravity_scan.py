from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone22 import run as prev
from milestone23.solver import FaceDamBreak

def main():
 prev.BalancedDamBreak=FaceDamBreak
 out={}
 for g in [.00075,.001,.0015,.002]:
  r,*_=prev.dam_case('gravity_scan','reference',background=True,gravity=g,steps=96);out[str(g)]=r
 (Path(__file__).parent/'results'/'gravity_scan.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
