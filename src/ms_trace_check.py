"""Check every decimal number in main.tex/supplement.tex against frozen source values (rounded match)."""
import re, glob, pandas as pd, numpy as np
U="/mnt/user-data/uploads/AI_training/P21_CTG_Acidemia/"
pool=[]
files=glob.glob(U+"outputs/t06b_synthesis/tables/*.csv")+glob.glob(U+"outputs/t09a_synthesis/tables/*.csv")+glob.glob(U+"outputs/t06d_figures/*.csv")+glob.glob(U+"outputs/t09a_synthesis/figures/*.csv")+[U+"reports/NUMBERS_TRACE.csv"]
for f in files:
    d=pd.read_csv(f,dtype=str)
    for v in d.values.ravel():
        for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?",str(v)):
            try: pool.append(float(x))
            except: pass
lit=[0.979,0.963,0.982,0.918,0.987,0.728,0.627,0.841,0.620,0.888,0.743,0.730,0.776,0.680,0.840]
pool=np.array(pool+lit)
derived={"5.8":30/518*100,"4.7":24/512*100,"25.0":25.0,"1.000":0.99966,"0.30":0.299,"0.29":0.2932,"0.10":0.104}
def ok(tok):
    v=float(tok); dec=len(tok.split(".")[1]) if "." in tok else 0
    for cand in (v,-v,v/100,v*100):
        if np.any(np.abs(np.round(pool,dec)-cand)<1e-9) or np.any(np.abs(np.round(pool*100,dec)-cand)<1e-9): return True
    return tok in derived
for fn in ["main.tex","supplement.tex","highlights.txt"]:
    s=re.sub(r"%.*","",open(fn).read())
    s=re.sub(r"\\(cite[pt]?|ref|label|includegraphics|SI|SIrange)(\[[^\]]*\])?\{[^}]*\}","",s)
    toks=re.findall(r"(?<![\w.])-?\d+\.\d+",s)
    bad=sorted(set(t for t in toks if not ok(t.lstrip("-"))))
    print(fn,"decimals:",len(toks),"unmatched:",bad)
