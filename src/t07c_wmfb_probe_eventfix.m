function t07c_wmfb_probe_eventfix()
% Read-only WMFB event-schema probe (MATLAB 3xN: start/end/peak).
% Does not generate .mat, metrics, or output files.
project = 'D:/Tuan/AI_training/P21_CTG_Acidemia';
repo = 'D:/Tuan/AI_training/data_raw/fhrma';
base = 'D:/Tuan/AI_training/data_proc/ctu-uhb-ctgdb/1.0.0/wmfb';
out = fullfile(project,'outputs','t07c_wmfb');
assert(exist(fullfile(out,'input_manifest.csv'),'file')==2,'T07C:ManifestMissing','Input manifest not found.');
assert(exist(fullfile(out,'official_eval.json'),'file')==2,'T07C:OfficialMissing','Official evaluation not found.');
addpath(genpath(repo));
baseDir=pwd; cleanup=onCleanup(@()cd(baseDir)); cd(repo);
try
    x=load(fullfile(repo,'FHRMAdataset','analyses','expertAnalyses.mat'),'data');
    matches=find(strcmpi({x.data.filename},'train01.fhr'));
    assert(numel(matches)==1,'T07C:TrainEntryMissing','train01 label entry not uniquely found.');
    ex=x.data(matches);
    [FHR1,FHR2,~,TOCO]=fhropen(fullfile(repo,'FHRMAdataset','traindata','train01.fhr'));
    [FHR,~,~]=preprocess(FHR1,FHR2,TOCO,ex.unreliableSignal);
    FHR=double(FHR(:)');
    fprintf('T07C_EVENTFIX_PROBE_TRAIN_START samples=%d finite=%d nonzero=%d\n',numel(FHR),all(isfinite(FHR)),nnz(FHR));
    [b,a,d]=aamwmfb(FHR);
    validate_output(b,a,d,numel(FHR),'train01');
    fprintf('T07C_EVENTFIX_PROBE_TRAIN_PASS baseline=%d acc=%d dec=%d\n',numel(b),size(a,2),size(d,2));

    opts=detectImportOptions(fullfile(out,'input_manifest.csv'));
    opts=setvartype(opts,'record_id','string');
    manifest=readtable(fullfile(out,'input_manifest.csv'),opts);
    assert(height(manifest)==552,'T07C:ManifestCount','Expected 552 records.');
    rid=char(manifest.record_id(1));
    inp=fullfile(base,'in',[rid '.mat']);
    raw=load(inp,'FHR0','fs','n_stage1');
    FHR0=double(raw.FHR0(:)');
    n=double(raw.n_stage1);
    assert(numel(FHR0)==n && raw.fs==4,'T07C:RawInvalid','Record sample count or sampling frequency invalid.');
    fprintf('T07C_EVENTFIX_PROBE_CTU_START rid=%s samples=%d valid=%d\n',rid,n,nnz(FHR0>0));
    assert(any(FHR0>0),'T07C:AllMissing','CTU record has no valid FHR.');
    FHR=interpolFHR(FHR0);
    [b,a,d]=aamwmfb(FHR);
    validate_output(b,a,d,n,rid);
    fprintf('T07C_EVENTFIX_PROBE_CTU_PASS rid=%s baseline=%d acc=%d dec=%d\n',rid,numel(b),size(a,2),size(d,2));
    fprintf('T07C_EVENTFIX_PROBE_PASS train=1 ctu=1 no_outputs_written=YES\n');
catch ME
    fprintf(2,'T07C_EVENTFIX_PROBE_FAIL id=%s message=%s\n',ME.identifier,ME.message);
    fprintf(2,'%s\n',getReport(ME,'extended','hyperlinks','off'));
    rethrow(ME);
end
end
function validate_output(b,a,d,n,rid)
assert(isnumeric(b) && numel(b)==n && all(isfinite(b(:))), ...
    'T07C:BaselineInvalid','WMFB baseline invalid for %s',rid);
validate_event(a,n,rid,'acc');
validate_event(d,n,rid,'dec');
end

function validate_event(ev,n,rid,kind)
% aamwmfb>adjustduration and validaccident retain peak in the third row.
% Empty event arrays may be 2x0 or 3x0 depending on detection path.
assert(isnumeric(ev) && ismatrix(ev) && all(isfinite(ev(:))), ...
    'T07C:EventNotNumeric','Invalid %s event values for %s',kind,rid);
if isempty(ev)
    assert(size(ev,2)==0 && (size(ev,1)==2 || size(ev,1)==3), ...
        'T07C:EventShape','Expected empty 2x0 or 3x0 %s array for %s, got %s',kind,rid,mat2str(size(ev)));
    return;
end
assert(size(ev,1)==3, 'T07C:EventShape', ...
    'Expected 3xN [start;end;peak] %s array for %s, got %s',kind,rid,mat2str(size(ev)));
assert(all(ev(1,:)>=0) && all(ev(2,:)>ev(1,:)) && all(ev(2,:)<=n/4+0.25), ...
    'T07C:EventBounds','%s interval outside signal for %s',kind,rid);
assert(all(ev(3,:)>=ev(1,:)) && all(ev(3,:)<=ev(2,:)), ...
    'T07C:EventPeakInvalid','%s peak outside event interval for %s',kind,rid);
end
