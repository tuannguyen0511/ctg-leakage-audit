import pandas as pd, numpy as np
d=pd.read_csv("raw.csv",dtype=str)
d["included_main"]="yes"; d["ctu_uhb_confirmed"]="yes"; d["director_check"]=""; d["director_correction"]=""
def fix(i,**kw):
    r=d.index[d.id==i][0]
    for k,v in kw.items(): d.at[r,k]=v
def add(i,col,txt): 
    r=d.index[d.id==i][0]; d.at[r,col]=(d.at[r,col]+"; " if d.at[r,col] else "")+txt
PROTO=["unit_of_split","split_scheme","oversampling","augmentation","model_selection","threshold_selection","signal_quality_exclusion"]
# rule 5: abstract_only -> protocol unclear
for i in d.id[d.access=="abstract_only"]:
    r=d.index[d.id==i][0]; changed=[c for c in PROTO if d.at[r,c]!="unclear"]
    for c in PROTO: d.at[r,c]="unclear"
    if changed: add(i,"director_correction","abstract_only rule: set to unclear: "+",".join(changed))
# eligibility
fix("S001",included_main="no",ctu_uhb_confirmed="no"); add("S001","director_correction","excluded: n=217 pre-release institutional set, not the public CTU-UHB")
fix("S004",included_main="no"); add("S004","director_correction","excluded from main: outcome = caesarean delivery (prompt exclusion rule); kept as leakage example (SMOTE before split)")
fix("S023",included_main="no"); add("S023","director_correction","excluded from main: unsupervised anomaly detection, outcome criterion not stated")
fix("S026",first_author="Liu",included_main="no",ctu_uhb_confirmed="unclear"); add("S026","director_correction","author per dblp key LiuLLBL21; CTU-UHB use not verified -> pending full text")
# verified corrections
fix("S025",unit_of_split="segment",units_per_record="~2-3 (20-min segments from last 60 min)"); add("S025","director_correction","image->segment: 106 abnormal segments from 44 records; 80/20 random split of segments, record grouping not stated")
fix("S027",oversampling="after_split"); add("S027","director_correction","oversampling before_split->after_split: 'GAN is used only for the training set'; model_selection on_test kept (test F1 shown in kernel-size selection, Fig 7)")
fix("S029",unit_of_split="unclear",oversampling="unclear",units_per_record="1 per analysis step (3 steps analysed separately)"); add("S029","director_correction","segment->unclear and before_split->unclear: three 30-min intervals analysed as separate experiments; timing of oversampling vs split not stated")
fix("S031",oversampling="after_split"); add("S031","director_correction","before_split->after_split: resampling drawn from training centres only (leave-one-centre-out)")
fix("S033",first_author="Chiou",doi_or_url="10.1038/s44294-025-00068-w"); add("S033","director_correction","author and published DOI from DOAJ record")
fix("S035",threshold_selection="unclear"); add("S035","director_correction","on_validation->unclear: threshold ~0.52 'determined using the CTU-UHB dataset only, based on cross-validation' (pooled OOF vs inner not stated)")
fix("S036",title="Fusing tabular features and deep learning for fetal heart rate analysis: a clinically interpretable model for fetal compromise detection")
add("S036","director_correction","title from University of Melbourne record")
# new study
new={c:"" for c in d.columns}
new.update(id="S037",first_author="Vargas-Calixto",year="2026",title="Self-Supervised Representation Learning of Fetal Heart Rate Outperforms a Foundation Model in Identifying Fetal Acidosis",venue="Computing in Cardiology 2026 (accepted abstract)",doi_or_url="https://cinc.org/2026/Program/accepted/19.html",access="abstract_only",model_family="DL",input="raw_signal",outcome_definition="fetal acidosis (definition not in abstract); CTU-UHB 29 acidosis cases",n_records_used="552 (external test)",units_per_record="unclear",external_validation="yes",external_dataset="CTU-UHB as external test; internal private cohort n=1217",reported_metric="AUROC",reported_value="CTU-UHB external: SSL 0.743; supervised 0.637; TimesFM 0.600",included_main="yes",ctu_uhb_confirmed="yes",director_correction="added by Director (missed by Deep Research)")
for c in PROTO: new[c]="unclear"
d=pd.concat([d,pd.DataFrame([new])],ignore_index=True)

# --- Director update 2: full texts supplied by Tuan (2026-11-13) ---
fix("S026",included_main="yes",ctu_uhb_confirmed="yes",access="full_text",outcome_definition="pH <= 7.15 abnormal (113/552)",n_records_used="552",units_per_record="1 (last 20 min)",unit_of_split="record",split_scheme="holdout",oversampling="none",augmentation="none",model_selection="on_test",threshold_selection="unclear",signal_quality_exclusion="unclear",reported_metric="SE SP QI",reported_value="SE 75.23% SP 70.82% QI 72.29% (mean of 10 random 90/10 splits)",quote_split="the total FHR signals were randomly divided into 10 segments and 90% (395 N and 101P) formed the training set while the remainder (10%, 44 N and 12P) was used to test",quote_selection="Based in the test set, we found that QI reaches the maximum value when the first convolutional layer kernel is 16x1",quote_resampling="Because of the data imbalance, we adjusted the weight of the classification",page_or_section="Sec 2.1, 3.1",title="An attention-based CNN-BiLSTM hybrid neural network enhanced with features of discrete wavelet transformation for fetal acidosis classification",venue="Expert Systems with Applications 186")
d.loc[d.id=="S026","director_correction"]="full text checked: CTU-UHB confirmed; record-level repeated holdout; kernel size and class weight tuned on test-set QI (on_test)"
d.loc[d.id=="S026","director_check"]="full text read (PDF)"
fix("S036",access="full_text",units_per_record="1 (last 60 min)",unit_of_split="record",split_scheme="cross_database",oversampling="none",augmentation="none",model_selection="separate_validation",threshold_selection="unclear",signal_quality_exclusion="no",exclusion_detail="all recordings included irrespective of signal loss; intermediate pH removed from training only",venue="IEEE Transactions on Biomedical Engineering 73(10)",reported_value="CTU-UHB external AUC 0.84 (fusion); FHR-only ResNet 0.80; internal 5-fold CV 0.77",quote_split="The MHW-pH data was split into training and validation datasets using 5-fold cross-validation. The CTU-UHB dataset was used as an external testing dataset.",quote_selection="FINAL MODEL SELECTED BASED ON 5-FOLD CV PERFORMANCE (Table II caption)",quote_exclusion="All FHR recordings of both datasets were included irrespective of their signal loss",page_or_section="Sec II-B, Table II")
d.loc[d.id=="S036","director_correction"]="full text checked: record-level external test; model chosen on internal CV (MHW-pH); no signal-loss exclusion"
d.loc[d.id=="S036","director_check"]="full text read (PDF)"

# verification status
ver={"S001":"DOI found (dblp, author PDF)","S002":"DOI found (dblp/unpaywall)","S003":"DOI found (PMC5686283)","S006":"DOI found (PMC6422985)","S007":"DOI found; full text blocked","S010":"title found (institutional repository)","S013":"DOI found (PubMed 31435480)","S014":"DOI found (Springer)","S015":"DOI found (MDPI)","S016":"DOI not confirmed (results page only)","S017":"paper found (dblp, author PDF); DOI not retrieved","S022":"DOI found; quotes checked (PMC9749291)","S023":"arXiv found","S025":"DOI found; quotes checked","S027":"DOI found; quotes checked (PMC10801938)","S028":"DOI found; quotes checked (PMC11202091)","S029":"DOI found; quotes checked (MDPI)","S030":"PubMed 38663662 found","S033":"DOI found (DOAJ); quotes checked (medRxiv)","S034":"arXiv found; quotes checked","S035":"DOI found; quotes checked (PMC13617899)","S037":"abstract checked (CinC 2026)","S000":""}
for k,v in ver.items(): d.loc[d.id==k,"director_check"]=v
d.loc[d.director_check=="","director_check"]="not individually checked (well-known DOI or seed list)"
# CTU-UHB AUROC used for summary
auc={"S006":0.987,"S007":0.978,"S019":0.63,"S020":0.68,"S024":0.743,"S025":0.980,"S028":0.918,"S030":0.72,"S031":0.74,"S032":0.81,"S033":0.62,"S034":0.826,"S035":0.888,"S036":0.84,"S037":0.743}
d["ctu_auroc_for_summary"]=d.id.map(auc)
def design(r):
    if r.external_validation=="yes" and r.split_scheme in("cross_database","unclear") and r.id not in("S035",): return "external_test_on_CTU"
    return {"record":"within_CTU_record","segment":"within_CTU_segment_or_image","image":"within_CTU_segment_or_image"}.get(r.unit_of_split,"within_CTU_unclear")
d["evaluation_design"]=d.apply(design,axis=1)
d.to_csv("U1_extraction.csv",index=False)
m=d[(d.included_main=="yes")]
out=[]
out.append(f"Main set: {len(m)} studies ({(m.access=='full_text').sum()} full text, {(m.access=='abstract_only').sum()} abstract only)")
f=m[m.access=="full_text"]
for c in ["unit_of_split","oversampling","augmentation","model_selection","threshold_selection"]:
    vc=f[c].value_counts(); out.append(f"{c} (full text, n={len(f)}): "+"; ".join(f"{k} {v} ({100*v/len(f):.0f}%)" for k,v in vc.items()))
out.append("\nCTU-UHB AUROC by evaluation design (main set, any access):")
for g,s in m.dropna(subset=["ctu_auroc_for_summary"]).groupby("evaluation_design"):
    a=s.ctu_auroc_for_summary.astype(float)
    out.append(f"  {g}: n={len(a)}, median {a.median():.3f} [IQR {a.quantile(.25):.3f}-{a.quantile(.75):.3f}], range {a.min():.3f}-{a.max():.3f}; ids {','.join(s.id)}")
open("U1_summary.txt","w").write("\n".join(out)+"\n"); print("\n".join(out))
