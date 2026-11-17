"""Synthetic preregistered invariants. Does not use CTU labels or PC-A data."""
import numpy as np
import pandas as pd
import pytest
from P21_CTG_Acidemia.src import t07d_featv2 as m
from P21_CTG_Acidemia.src import t03_baselines as t03
from P21_CTG_Acidemia.src import t05_leakage as t05


def test_prereg_and_exact_11():
    assert m.PREREG=='df06f9dd107ffb5b4132af3ade17aa9fec6c71d3f0b8154803c8ddfc3c59a557'
    assert len(m.COLUMNS)==11 and len(m.NEW)==5 and len(m.OLD)==6
    assert set(m.COLUMNS)&set(m.OLD)==set(m.OLD)
    assert not any('pH' in c or 'y705' in c for c in m.COLUMNS)


def test_midpoint_count_prolonged_missing_and_area():
    fs=4;n=800 # 200 s
    fhr=np.full(n,130.,dtype=float)
    base=np.full(n,140.,dtype=float)
    fhr[80:720]=120.
    fhr[100:104]=np.nan
    events=[
        {'type':'dec','start_s':20.,'end_s':180.,'flag_missing':False},
        {'type':'acc','start_s':5.,'end_s':15.,'flag_missing':False},
        {'type':'dec','start_s':110.,'end_s':140.,'flag_missing':True},
        {'type':'acc','start_s':175.,'end_s':195.,'flag_missing':False},
    ]
    x=m.wmfb_only(base,events,fhr,0,n)
    assert x['baseline_v2']==140
    assert x['n_accel_v2']==2 and x['n_decel_v2']==1
    assert x['n_prolonged_decel_v2']==1
    expected=(636*20)*0.25 # 640 samples minus 4 missing
    assert x['decel_area_v2']==expected


def test_midpoint_window_boundary_and_area_intersection():
    base=np.full(800,140.,dtype=float);fhr=np.full(800,130.,dtype=float)
    events=[{'type':'dec','start_s':50,'end_s':150,'flag_missing':False}]
    left=m.wmfb_only(base,events,fhr,0,400)  # midpoint 100s excluded exactly
    right=m.wmfb_only(base,events,fhr,400,800)
    assert left['n_decel_v2']==0 and left['decel_area_v2']==0
    assert right['n_decel_v2']==1 and right['decel_area_v2']==500.


def test_decel_area_clips_at_zero():
    b=np.full(80,120.,dtype=float); f=np.full(80,150.)
    e=[{'type':'dec','start_s':0,'end_s':20,'flag_missing':False}]
    assert m.wmfb_only(b,e,f,0,80)['decel_area_v2']==0.


def test_no_events():
    f=np.full(80,120.)
    r=m.wmfb_only(f,[],f,0,80)
    assert r['n_accel_v2']==r['n_decel_v2']==0


def test_baseline_invalid_fails():
    b=np.full(100,140.);b[2]=np.nan
    with pytest.raises(RuntimeError,match='WMFB_NONFINITE_BASELINE'):
        m.wmfb_only(b,[],np.full(100,120.),0,100)


def test_invalid_bounds_fails():
    with pytest.raises(RuntimeError,match='INVALID_WINDOW'):
        m.wmfb_only(np.ones(50),[],np.ones(50),50,50)


def test_no_flagged_events_counted():
    base=np.full(800,140.)
    x=m.wmfb_only(base,[{'type':'acc','start_s':0,'end_s':10,'flag_missing':True},
                         {'type':'dec','start_s':20,'end_s':130,'flag_missing':True}],
                  np.full(800,120.),0,800)
    assert x['n_accel_v2']==0 and x['n_decel_v2']==0 and x['decel_area_v2']==0


def test_six_original_values_preserved_and_column_order():
    original=pd.DataFrame({'record_id':['01','02'],**{
        c:[1.,np.nan] if c=='stv_ms' else [2.,3.] for c in t03.FEATURES}})
    fresh=pd.DataFrame({'record_id':['01','02'],**{c:[10.,20.] for c in m.NEW}})
    r=m.assemble_v2(original,fresh,'record_id')
    assert list(r)==['record_id',*m.COLUMNS]
    for c in m.OLD:
        assert np.array_equal(np.isnan(r[c]),np.isnan(original[c]))
        assert np.allclose(r[c],original[c],equal_nan=True)
    for c in m.NEW:assert list(r[c])==[10,20]


def test_units_grid_and_nested_t03_t05_reuse():
    units=list(m.all_units())
    assert len(units)==160 and len(set(units))==160
    assert sum(u[0]=='S' for u in units)==100
    for p in ('S_seg','L1','L2','L3'):
        assert sum(u[0]==p for u in units)==15
    assert t03.grid('M0',{'m0_c':[.01,.1,1.,10.]})==[{'C':.01},{'C':.1},{'C':1.},{'C':10.}]


def test_t05_segment_offsets_direct_import():
    fhr=np.full(4*60*20,120.)
    uc=fhr.copy()
    x,d=t05.stage1_segments(fhr,uc,4,10,5,50)
    assert [a for a,b,c in x]==[0,1200,2400]
    assert d==0


def test_t05_segment_missing_rejected():
    f=np.full(2400,120.);f[:1201]=np.nan
    x,d=t05.stage1_segments(f,f,4,10,5,50)
    assert not x and d==1


def test_strict_oof_unique_each_repeat():
    ids=['001','002','003']
    df=pd.DataFrame([{'record_id':rid,'repeat':r,'y':int(i==0),
                      'p_M0':0.5,'p_M1':0.5}
                      for i,rid in enumerate(ids) for r in range(10)])
    m.validate_oof(df,ids,10,['p_M0','p_M1'])
    with pytest.raises(RuntimeError,match='OOF_ROW_COUNT_MISMATCH'):
        m.validate_oof(df.iloc[:-1],ids,10,['p_M0','p_M1'])


def test_strict_oof_nan_fails():
    df=pd.DataFrame({'record_id':['1'],'repeat':[0],'y':[0],'p_M0':[np.nan]})
    with pytest.raises(RuntimeError,match='OOF_NONFINITE'):
        m.validate_oof(df,['1'],1,['p_M0'])


def test_record_alignment_not_position_order():
    a=pd.DataFrame({'y':[1,0],'p':[0.8,0.2]},index=['A','B'])
    b=pd.DataFrame({'y':[0,1],'p':[0.1,0.9]},index=['B','A'])
    y,x,z=m.align_two(a,b)
    assert list(y)==[1,0] and list(x)==[.8,.2] and list(z)==[.9,.1]


def test_validated_wmfb_event_schema_and_flag():
    e=pd.DataFrame({'record_id':['001'],'type':['dec'],'start_s':[4.],
                    'end_s':[10.],'pct_missing_raw':[51.],'flag_missing':[True]})
    m.validate_events(e,{'001':100})
    e['flag_missing']=False
    with pytest.raises(RuntimeError,match='FLAG_MISSING_INCONSISTENT'):
        m.validate_events(e,{'001':100})


def test_lock_lookups_require_context():
    rows=[('outputs/t03_baselines/a_primary/oof.parquet 1a2b3c4d','1a2b3c4d'),
          ('outputs/t03_baselines/b_noexcl/oof.parquet deadbeef','deadbeef')]
    assert m.lock_entry(rows,'oof.parquet','a_primary/oof.parquet')=='1a2b3c4d'
    with pytest.raises(RuntimeError,match='FROZEN_LOCK_MISSING_OR_AMBIGUOUS'):
        m.lock_entry(rows,'oof.parquet')


def test_full_comparison_path_without_real_data(monkeypatch,tmp_path):
    """Exercise model contrasts, leakage units, record pairing, E2 (B=20)."""
    rng=np.random.default_rng(44)
    ids=np.asarray([f'{i:04d}' for i in range(518)])
    y=np.r_[np.ones(30,dtype=int),np.zeros(488,dtype=int)]
    rng.shuffle(y)
    folds=np.tile(np.arange(5),104)[:518]
    strict=pd.DataFrame([{'record_id':rid,'repeat':r,'outer_fold':int(folds[ix]),
         'y':int(y[ix]),'p_M0':float(rng.uniform(.2,.8)),
         'p_M1':float(rng.uniform(.2,.8))}
         for r in range(10) for ix,rid in enumerate(ids)])
    old=strict.copy()
    old[['p_M0','p_M1']]=rng.uniform(.2,.8,(len(old),2))
    d04=strict[['record_id','repeat','outer_fold','y']].copy()
    for model in ('M2','M3'):
        for seed in range(3):d04[f'p_{model}_s{seed}']=rng.uniform(.2,.8,len(d04))
    segid=np.asarray([f'{rid}__{i:08d}' for rid in ids for i in (0,1200)])
    segrec=np.repeat(ids,2);segy=np.repeat(y,2)
    def segframe():
        return pd.DataFrame([{'row_id':row,'record_id':rid,'repeat':rep,'outer_fold':rep%5,
             'model':'M1','seed':0,'y':int(label),'p':float(rng.uniform(.1,.9))}
             for rep in range(3) for row,rid,label in zip(segid,segrec,segy)])
    sseg=segframe(); l1=segframe()
    rows2=[('real::'+rid,rid,int(label)) for rid,label in zip(ids,y)]
    rows2 += [(f'smote::{j:05d}',f'smote::{j:05d}',1) for j in range(458)]
    l2=pd.DataFrame([{'row_id':row,'record_id':rid,'y':label,'repeat':rep,
                      'outer_fold':rep%5,'seed':0,'model':'M1','p':float(rng.uniform(.1,.9))}
                      for rep in range(3) for row,rid,label in rows2])
    l3=pd.DataFrame([{'row_id':rid,'record_id':rid,'y':int(label),'repeat':rep,
                     'outer_fold':rep%5,'seed':0,'model':'M1','p':float(rng.uniform(.1,.9))}
                     for rep in range(3) for rid,label in zip(ids,y)])
    fake={
        't03_baselines/A_primary/oof.parquet':old,
        't04_dl/A_primary/oof.parquet':d04,
        't05_leakage/oof_L1.parquet':l1,
        't05_leakage/oof_S_seg.parquet':sseg,
    }
    original_read_parquet=pd.read_parquet
    def read_parquet(path,*args,**kwargs):
        path=str(path).replace('\\','/')
        for tail,df in fake.items():
            if path.endswith(tail):return df.copy()
        raise AssertionError('Unexpected parquet read '+path)
    def read_csv(path,*args,**kwargs):
        if str(path).replace('\\','/').endswith('t05_leakage/h2_tests.csv'):
            return pd.DataFrame([{'protocol':p,'model':'M1','delta':.01,
              'ci95_low':-.03,'ci95_high':.06,'p_raw':.5,'bootstrap_valid':20}
              for p in ('L1','L2','L3')])
        raise AssertionError('Unexpected CSV read '+str(path))
    monkeypatch.setattr(pd,'read_parquet',read_parquet)
    monkeypatch.setattr(pd,'read_csv',read_csv)
    labelvals=np.linspace(0,1,518)
    rng.shuffle(labelvals)
    labels=pd.DataFrame({'fhr_nan_pct_w30':labelvals},index=ids)
    data={'ids':ids,'y':y,'labels':labels}
    segments={'row_id':segid}
    result,details=m.compare_results({'bootstrap_resamples':20,'bootstrap_seed':42},
        data,segments,{'S':strict,'S_seg':sseg,'L1':l1,'L2':l2,'L3':l3},tmp_path)
    assert len(result)==14
    assert result['p_raw'].between(0,1).all()
    assert set(details['E2_v2'])=={'le_median','gt_median'}
    assert 'record' in details['original_T05_leakage']['L1']
    assert len(details['model_metrics'])==6
