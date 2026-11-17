"""Synthetic, no-training acceptance tests for P21-T06b."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from P21_CTG_Acidemia.src import t06b_synthesis as m


def test_parse_lock_t04_style():
    text = ('62a0da0e outputs/t04_dl/A_primary/oof.parquet\n'
            'af48a9c9 outputs/t04_dl/A_primary/metrics.json\n')
    parsed = m.parse_lock_lines(text, None)
    assert parsed['P21_CTG_Acidemia/outputs/t04_dl/A_primary/oof.parquet'] == '62a0da0e'
    assert parsed['P21_CTG_Acidemia/outputs/t04_dl/A_primary/metrics.json'] == 'af48a9c9'


def test_parse_lock_sha256_and_conflict():
    s='a'*64
    assert m.parse_lock_lines(s+' *outputs/t04_dl/A_primary/oof.parquet',None)[
        'outputs/t04_dl/A_primary/oof.parquet']==s
    with pytest.raises(ValueError,match='Conflicting'):
        m.parse_lock_lines('abcd1234 outputs/x\n1234abcd outputs/x',None)


def test_missing_or_wrong_frozen_sha(tmp_path):
    d=tmp_path/'P21_CTG_Acidemia'/'reports';d.mkdir(parents=True)
    inp=tmp_path/'data_proc'/'test.txt';inp.parent.mkdir();inp.write_text('input')
    with pytest.raises(ValueError,match='No .*FROZEN_SHA'):
        m.verify_locked_inputs(tmp_path,[inp])
    (d/'T02_FROZEN_SHA.txt').write_text('00000000 data_proc/test.txt\n')
    with pytest.raises(ValueError,match='FROZEN_SHA_MISMATCH'):
        m.verify_locked_inputs(tmp_path,[inp])
    (d/'T02_FROZEN_SHA.txt').write_text(m.sha(inp)[:8]+' data_proc/test.txt\n')
    result, files=m.verify_locked_inputs(tmp_path,[inp]);assert len(result)==1 and len(files)==1


def test_suppress_counts_and_triplets():
    assert m.suppress_count(0)=='0'
    assert m.suppress_count(9)=='<10'
    assert m.suppress_count(10)=='10'
    assert m.stats_triplet(pd.Series([1.,2.,3.]))==(2.,1.5,2.5)


def test_exact_trace_from_csv_and_nested_json(tmp_path):
    p=tmp_path/'sample.csv'
    pd.DataFrame([dict(protocol='L4',model='M1',delta='0.15',
                       details=json.dumps({'mean_fold_Se_S':0.25,'ci95_s_record':[0.4,0.6]}))]).to_csv(p,index=False)
    t=[]
    m.trace_add(t,'T4','L4/M1','delta','0.15',p,'delta',tmp_path)
    m.trace_add(t,'T4','L4/M1','S_se',0.25,p,'details.mean_fold_Se_S',tmp_path)
    m.trace_add(t,'T4','L4/M1','CI_low',0.4,p,'details.ci95_s_record[0]',tmp_path)
    assert m.audit_t2_t5_exact(t,tmp_path)==3
    t[0]['value']='0.1500001'
    with pytest.raises(ValueError,match='EXACT_MATCH_FAIL'):
        m.audit_t2_t5_exact(t,tmp_path)


def make_synthetic_root(tmp_path,monkeypatch):
    root=tmp_path
    base=root/'P21_CTG_Acidemia'
    report=base/'reports';report.mkdir(parents=True)
    derived=root/'data_proc/ctu-uhb-ctgdb/1.0.0';derived.mkdir(parents=True)
    ids=[f'fake_{i:04d}' for i in range(552)]
    y=np.array(([1]*40)+([0]*512),dtype=int)
    excluded=np.array(([True]*10)+([False]*30)+([True]*24)+([False]*488))
    assert excluded.sum()==34 and y[~excluded].sum()==30
    labels=pd.DataFrame({'record_id':ids,'y705':y,'excluded':excluded,
                         'no_stage2':np.array([False]*520+[True]*32),
                         'fhr_nan_pct_w30':np.linspace(0,40,552)})
    clinical=pd.DataFrame({'record_name':ids,'pH':np.where(y==1,7.0,7.2),
                           'BDecf':np.linspace(0,10,552)})
    qc=pd.DataFrame({'record_name':ids,'fhr_missing_pct':np.zeros(552)})
    features=pd.DataFrame({'record_id':ids,'baseline':np.repeat(140.,552)})
    data_map={'labels.parquet':labels,'clinical.parquet':clinical,
              'signal_qc.parquet':qc,'features_w30.parquet':features}
    base_pred=np.linspace(.005,.995,518)
    eligible_ids=np.array(ids)[~excluded]
    eligibility_y=y[~excluded]
    out3=[];out4=[]
    for repeat in range(10):
        for i,rid in enumerate(eligible_ids):
            yy=int(eligibility_y[i]);fold=i%5
            out3.append({'record_id':rid,'repeat':repeat,'outer_fold':fold,'y':yy,
                         'p_M0':float(base_pred[i]),'p_M1':float(np.clip(base_pred[i]*.8+.1,0,1))})
            row={'record_id':rid,'repeat':repeat,'outer_fold':fold,'y':yy}
            for model in ('M2','M3'):
                for seed in range(3):
                    row[f'p_{model}_s{seed}']=float(np.clip(base_pred[i]*.7+.14+seed*.005,0,1))
            out4.append(row)
    frames={str(base/'outputs/t03_baselines/A_primary/oof.parquet'):pd.DataFrame(out3),
            str(base/'outputs/t04_dl/A_primary/oof.parquet'):pd.DataFrame(out4),
            str(base/'outputs/t05_leakage/segments_meta.parquet'):pd.DataFrame({
                'n_segments':[5224]+[0]*517,'n_dropped_missing':[499]+[0]*517})}
    for f,df in data_map.items():
        frames[str(derived/f)]=df
    for f in m.input_paths(root):
        f.parent.mkdir(parents=True,exist_ok=True)
        if f.suffix=='.parquet':f.write_bytes(b'fake-parquet-bytes')
    for config in m.CONFIGS:
        t3={}; t4={'models':{}}
        for model in ('M0','M1'):
            met={'auroc':.6,'auprc_average_precision':.1,'brier':.2,
                 'calibration_intercept_offset_logit':.4,'calibration_slope':.7,
                 'sensitivity_at_specificity_0_85':.5}
            t3[model]={'mean_probability_metrics':met,
                       'auroc_ci95':[.5,.7],'auprc_ci95':[.08,.14]}
        for model in ('M2','M3')+(('M2UC',) if config=='A_primary' else ()):
            met={'auroc':.7,'auroc_ci95':[.6,.8],
                 'auprc_average_precision':.12,'auprc_ci95':[.08,.17],
                 'brier':.3,'calibration_intercept_offset_logit':.5,'calibration_slope':.6,
                 'sensitivity_at_specificity_0_85':.7}
            t4['models'][model]={'mean_probability_over_repeats_and_seeds':met}
        (base/f'outputs/t03_baselines/{config}/metrics.json').write_text(json.dumps(t3))
        (base/f'outputs/t04_dl/{config}/metrics.json').write_text(json.dumps(t4))
    h1=[{'config':'A_primary','comparison':x,'delta_auroc':.1,'ci95_low':.01,
         'ci95_high':.2,'p_raw':.02,'p_holm':.04} for x in ('M2-M1','M3-M1')]
    pd.DataFrame(h1).to_csv(base/'outputs/t06a_stats/h1_tests.csv',index=False)
    sens=[{'config':c,'comparison':mo+'-M1','delta_auroc':.1,'ci95_low':.01,
           'ci95_high':.2,'p_raw':.04} for c in m.CONFIGS[1:] for mo in ('M2','M3')]
    pd.DataFrame(sens).to_csv(base/'outputs/t06a_stats/sensitivity.csv',index=False)
    exploratory=[]
    groups={'E1':['fhr_nan_pct_w30']*2,
            'E2':[f'{g}_{suffix}' for g in ('le_median','gt_median') for suffix in ('M1','M3','M3-M1')],
            'E3':['M1_plus_fhr_nan_pct_w30','M1_plus_missingness-M1','M1_plus_missingness-M3'],
            'E4':[f'{a}_{mo}' for a in ('all','y705_eq_0') for mo in ('M1','M2','M3')]}
    for analysis,gs in groups.items():
        for i,g in enumerate(gs):
            exploratory.append({'analysis':analysis,'config':'B_noexcl' if analysis=='E1' and i==1 else 'A_primary',
                                'group':g,'n':518,'positives':'30' if not g.startswith('le_') else '<10',
                                'measure':'AUROC' if analysis!='E4' else 'Spearman_rho',
                                'estimate':.6,'ci95_low':.5,'ci95_high':.7})
    assert len(exploratory)==17
    pd.DataFrame(exploratory).to_csv(base/'outputs/t06a_stats/exploratory.csv',index=False)
    rows=[]
    for protocol in ('L1','L2','L3','L4'):
        for model in ('M1','M2'):
            details=({'mean_fold_Se_S':.4,'mean_fold_Sp_S':.7,'mean_fold_Se_L':.6,
                      'mean_fold_Sp_L':.65} if protocol=='L4' else
                     {'auroc_s_segment':.6,'auroc_l_segment':.7,
                      'ci95_s_segment':[.5,.7],'ci95_l_segment':[.6,.8],
                      'auroc_s_record':.61,'auroc_l_record':.75,
                      'ci95_s_record':[.51,.71],'ci95_l_record':[.65,.85]})
            if protocol=='L2':
                details.update(auroc_l_all_test=.99,delta_l_all_test_descriptive_only=.35)
            rows.append({'protocol':protocol,'model':model,'contrast':protocol+'-S',
                         'metric':'delta_Se_record_cluster' if protocol=='L4' else 'delta_AUROC',
                         'delta':.1,'ci95_low':.02,'ci95_high':.2,'p_raw':.002,
                         'p_holm':.01,'bootstrap_valid':10000,'seed_deltas':'[0.1,0.1,0.1]',
                         'details':json.dumps(details)})
    pd.DataFrame(rows).to_csv(base/'outputs/t05_leakage/h2_tests.csv',index=False)
    # source sha locks on every read input, including dummy parquet files.
    (report/'T06B_TEST_FROZEN_SHA.txt').write_text('\n'.join(
        m.sha(p)[:8]+' '+m.rel(p,root) for p in m.input_paths(root))+'\n')
    oldread=pd.read_parquet
    def fake_read(path,*args,**kwargs):
        return frames[str(path)] if str(path) in frames else oldread(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',fake_read)
    yamlfile=root/'test.yaml'
    yamlfile.write_text(json.dumps({'task':'P21-T06b','run_name':'t06b_synthesis',
                                    'prereg_sha256':m.PREREG,'work_root':str(root)}))
    return root,base,yamlfile


def test_synthetic_complete_no_training(tmp_path,monkeypatch):
    root,base,yamlfile=make_synthetic_root(tmp_path,monkeypatch)
    assert m.run(yamlfile,'inspect')==0
    assert not (base/'outputs/t06b_synthesis').exists()
    assert m.run(yamlfile,'run')==0
    output=base/'outputs/t06b_synthesis'
    assert (output/'summary.json').exists()
    assert len(list((output/'tables').glob('T*.csv')))==5
    assert len(list((output/'tables').glob('T*.tex')))==5
    assert len(list((output/'figures').glob('F*.png')))==5
    assert len(list((output/'figures').glob('F*.svg')))==5
    traces=pd.read_csv(base/'reports/NUMBERS_TRACE.csv')
    assert traces.source_sha8.notna().all() and len(traces)>150
    summary=json.loads((output/'summary.json').read_text())
    assert summary['source_numeric_values_checked']>100
    with pytest.raises(ValueError,match='output directory exists'):
        m.run(yamlfile,'run')


def test_sha8_and_full_hash_prefix_consistency():
    full = 'abcdef01' + '0'*56
    text = f'abcdef01 outputs/x\n{full} outputs/x\n'
    parsed=m.parse_lock_lines(text,None)
    assert parsed['outputs/x']==full


def test_renders_booktabs_and_escaped_names(tmp_path):
    m.emit_table(tmp_path,'T1_test',pd.DataFrame([{'name':'x_y','n':'<10','pct':'5%'}]))
    out=(tmp_path/'T1_test.tex').read_text()
    assert r'\toprule' in out and r'\bottomrule' in out
    assert r'x\_y' in out and r'5\%' in out


def test_strict_source_immutability_and_output_manifest(tmp_path,monkeypatch):
    root,base,cfg=make_synthetic_root(tmp_path,monkeypatch)
    inputs=m.input_paths(root)
    before=[m.sha(p) for p in inputs]
    assert m.run(cfg,'run')==0
    after=[m.sha(p) for p in inputs]
    assert before==after
    out=base/'outputs/t06b_synthesis'
    manifest=out/'OUTPUT_SHA256.txt'
    assert manifest.exists()
    for line in manifest.read_text().splitlines():
        expected, filename=line.split('  ',1)
        assert m.sha(root/filename)==expected
    summary=json.loads((out/'summary.json').read_text())
    assert all(w>=2000 for w,h in summary['figures_pixels'].values())
