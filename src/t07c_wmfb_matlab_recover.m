function t07c_wmfb_matlab_recover()
% P21-T07c: call unmodified FHRMA MATLAB routines, train and CTU stage 1.
% Run only after t07c_wmfb_probe PASS. Existing input and official evaluations reused.
% No toolbox files are written. Existing valid record outputs are reused.

project = 'D:/Tuan/AI_training/P21_CTG_Acidemia';
repo = 'D:/Tuan/AI_training/data_raw/fhrma';
base = 'D:/Tuan/AI_training/data_proc/ctu-uhb-ctgdb/1.0.0/wmfb';
out = fullfile(project,'outputs','t07c_wmfb');
assert(exist(fullfile(out,'input_manifest.csv'),'file')==2, 'T07C:ExportMissing', 'T07C check failed: ExportMissing');
assert(exist(fullfile(repo,'FHRMAdataset','analyses','expertAnalyses.mat'),'file')==2, 'T07C:ExpertMissing', 'T07C check failed: ExpertMissing');
assert(exist(fullfile(out,'provenance.json'),'file')==2, 'T07C:ProvenanceMissing', 'T07C check failed: ProvenanceMissing');
addpath(genpath(repo));
assert(exist('aamwmfb','file')==2 && exist('evaluateaam','file')==2 && exist('interpolFHR','file')==2,...
       'T07C:MATLABToolboxUnavailable', 'T07C check failed: MATLABToolboxUnavailable');
assert(exist('decimate','file')~=0 && exist('filtfilt','file')~=0,...
       'T07C:SignalProcessingToolboxRequired', 'T07C check failed: SignalProcessingToolboxRequired');
originalDir = pwd;
back = onCleanup(@()cd(originalDir));
cd(repo); % evaluateaam uses relative paths in the official FHRMA toolbox.

t0matlab=tic;
if exist(fullfile(out,'matlab_runtime_recovery.json'),'file')==2
    error('T07C:RecoveredAlreadyFinished','Recovery runtime file exists. Refuse rerun.');
end
if exist(fullfile(out,'matlab_errors_recovery'),'dir')~=7
    mkdir(fullfile(out,'matlab_errors_recovery'));
end
% STEP 1: Official evaluation. It adds a MEDIAN summary row at index 1.
officialFile=fullfile(out,'official_eval.json');
if exist(officialFile,'file')~=2
    assert(exist(fullfile(out,'official_fileresults.mat'),'file')~=2,'T07C:PartialOfficialOutputRequiresAudit', 'T07C check failed: PartialOfficialOutputRequiresAudit');
    try
        expert=load(fullfile(repo,'FHRMAdataset','analyses','expertAnalyses.mat'),'data');
        train=expert.data([expert.data(:).trainingData]==1);
        assert(numel(train)==66,'T07C:TrainingRecordCountChanged', 'T07C check failed: TrainingRecordCountChanged');
        fprintf('T07C_OFFICIAL_EVALUATEAAM_START n=66\n');
        % Do not replace or reimplement this official function.
        result=evaluateaam('aamwmfb(FHR)',train,'FHRMAdataset/traindata/');
        assert(numel(result)==67,'T07C:OfficialResultCountInvalid', 'T07C check failed: OfficialResultCountInvalid');
        assert(strcmp(char(result(1).filename),'Median'),'T07C:MissingOfficialMedianRow', 'T07C check failed: MissingOfficialMedianRow');
        fieldList=fieldnames(result);
        med=struct();
        for j=1:numel(fieldList)
            nm=fieldList{j};
            if strcmp(nm,'filename'), continue; end
            vals=[result(2:end).(nm)]; % do not include row 1 summary twice.
            if ~isnumeric(vals), continue; end
            vals=double(vals(:));
            if all(isfinite(vals))
                med.(nm)=median(vals);
            else
                % Record nonfinite fields explicitly, not invented zero.
                med.(nm)=NaN;
            end
        end
        assert(isfield(med,'MADI') && isfield(med,'RMSD_bpm'), 'T07C:RequiredOfficialFieldsMissing', 'T07C check failed: RequiredOfficialFieldsMissing');
        assert(isfinite(med.MADI) && isfinite(med.RMSD_bpm), 'T07C:OfficialPrimaryNonfinite', 'T07C check failed: OfficialPrimaryNonfinite');
        summary=struct('status','PASS','train_records',66,'median_metrics',med,...
                       'fieldnames',{fieldList},'first_row_is_median',true,...
                       'toolbox_command','evaluateaam(''aamwmfb(FHR)'',train,''FHRMAdataset/traindata/'')');
        save(fullfile(out,'official_fileresults.mat'),'result','-v7');
        safe_write_text(officialFile,jsonencode(summary));
        fprintf('T07C_OFFICIAL_EVALUATEAAM_PASS MADI=%.8g RMSD=%.8g fields=%d\n',...
                med.MADI,med.RMSD_bpm,numel(fieldList)-1);
    catch ME
        safe_write_error(fullfile(out,'matlab_errors_recovery','official_evaluateaam.txt'),getReport(ME,'extended','hyperlinks','off'));
        fprintf(2,'T07C_OFFICIAL_EVALUATEAAM_FAIL %s\n',ME.message);
        rethrow(ME);
    end
else
    o=jsondecode(fileread(officialFile));
    assert(exist(fullfile(out,'official_fileresults.mat'),'file')==2,'T07C:OfficialSourceResultsMissing', 'T07C check failed: OfficialSourceResultsMissing');
    assert(strcmp(o.status,'PASS') && o.train_records==66,'T07C:OfficialResultInvalid', 'T07C check failed: OfficialResultInvalid');
    fprintf('T07C_OFFICIAL_EVALUATEAAM_REUSE\n');
end

% STEP 1 (continued): same FHR preprocessing and predictions as the official evaluator.
trainRoot=fullfile(repo,'FHRMAdataset','traindata');
exp=load(fullfile(repo,'FHRMAdataset','analyses','expertAnalyses.mat'),'data');
train=exp.data([exp.data(:).trainingData]==1);
trainFailures=0;
for k=1:numel(train)
    item=train(k);
    rid=char(item.filename);
    [~,stem,ext]=fileparts(rid);
    assert(strcmpi(ext,'.fhr'),'T07C:InvalidFHRFilename', 'T07C check failed: InvalidFHRFilename');
    dest=fullfile(out,'train_pred',[stem '.mat']);
    if exist(dest,'file')==2
        assert_valid_mat(dest,[]);
        continue;
    end
    try
        [FHR1,FHR2,~,TOCO]=fhropen(fullfile(trainRoot,rid));
        [FHR,FHRraw,TOCO]=preprocess(FHR1,FHR2,TOCO,item.unreliableSignal); %#ok<ASGLU>
        FHR=double(FHR(:)');
        assert(numel(FHR)==numel(FHR1) && all(isfinite(FHR)), 'T07C:TrainFHRInvalid', 'T07C check failed: TrainFHRInvalid');
        [baseline,accelerations,decelerations]=aamwmfb(FHR);
        n_stage1=numel(FHR);
        assert_pred(baseline,accelerations,decelerations,n_stage1);
        save(dest,'baseline','accelerations','decelerations','n_stage1','-v7');
    catch ME
        trainFailures=trainFailures+1;
        safe_write_error(fullfile(out,'matlab_errors_recovery',['train_' stem '.txt']),getReport(ME,'extended','hyperlinks','off'));
        fprintf(2,'T07C_TRAIN_FAIL %s : %s %s\n',stem,ME.identifier,ME.message);
        if trainFailures>=3 && numel(dir(fullfile(out,'train_pred','*.mat')))==0
            error('T07C:TrainRepeatedFailure','First three train records failed. Review matlab_errors_recovery before continuing.');
        end
    end
end
fprintf('T07C_WMFB_TRAIN_DONE files=%d failures=%d\n',numel(dir(fullfile(out,'train_pred','*.mat'))),trainFailures);

% STEP 2: CTU signals extracted by Python with stage-2 cutoff, no pH data.
opts=detectImportOptions(fullfile(out,'input_manifest.csv'));
opts=setvartype(opts,'record_id','string');
manifest=readtable(fullfile(out,'input_manifest.csv'),opts);
assert(height(manifest)==552,'T07C:InputRecordCountChanged', 'T07C check failed: InputRecordCountChanged');
failures=0;completed=0;skipped=0;
for i=1:height(manifest)
    rid=char(manifest.record_id(i));
    input=fullfile(base,'in',[rid '.mat']);
    dest=fullfile(base,'out',[rid '.mat']);
    if exist(dest,'file')==2
        assert_valid_mat(dest,double(manifest.n_stage1(i)));
        skipped=skipped+1;
        continue;
    end
    try
        raw=load(input,'FHR0','fs','n_stage1');
        assert(raw.fs==4,'T07C:UnexpectedSamplingRate', 'T07C check failed: UnexpectedSamplingRate');
        n_stage1=double(raw.n_stage1);
        assert(n_stage1==double(manifest.n_stage1(i)), 'T07C:SourceLengthChanged', 'T07C check failed: SourceLengthChanged');
        FHR0=double(raw.FHR0(:)');
        assert(numel(FHR0)==n_stage1 && all(isfinite(FHR0)),'T07C:RawFHRInvalid', 'T07C check failed: RawFHRInvalid');
        assert(any(FHR0>0),'T07C:NoValidFHR', 'T07C check failed: NoValidFHR');
        FHR=interpolFHR(FHR0); % no T02 filtering/cleaning
        assert(numel(FHR)==n_stage1 && all(isfinite(FHR)),'T07C:InterpolatedFHRInvalid', 'T07C check failed: InterpolatedFHRInvalid');
        [baseline,accelerations,decelerations]=aamwmfb(FHR);
        assert_pred(baseline,accelerations,decelerations,n_stage1);
        save(dest,'baseline','accelerations','decelerations','n_stage1','-v7');
        completed=completed+1;
    catch ME
        failures=failures+1;
        safe_write_error(fullfile(out,'matlab_errors_recovery',[rid '.txt']),getReport(ME,'extended','hyperlinks','off'));
        fprintf(2,'T07C_CTU_MATLAB_FAIL %s: %s %s\n',rid,ME.identifier,ME.message);
        if failures>=3 && completed==0 && skipped==0
            error('T07C:CTURepeatedFailure','First three CTU records failed. Review matlab_errors_recovery.');
        end
    end
    if mod(i,25)==0
        fprintf('T07C_CTU_PROGRESS=%d/552 new=%d reused=%d fail=%d\n',i,completed,skipped,failures);
    end
end
matlab_run=struct('matlab_runtime_s',toc(t0matlab),'ctu_new',completed,...
   'ctu_reused',skipped,'ctu_failed',failures,'train_failed',trainFailures);
safe_write_text(fullfile(out,'matlab_runtime_recovery.json'),jsonencode(matlab_run));
fprintf('T07C_MATLAB_DONE completed=%d reused=%d failed=%d official=PASS train_failed=%d runtime_s=%.2f\n',...
         completed,skipped,failures,trainFailures,matlab_run.matlab_runtime_s);
if trainFailures>0 || failures>0
    error('T07C:WMFBIncomplete','Recovery finished with %d train and %d CTU failures; do not finalize without review.',trainFailures,failures);
end
end

function assert_pred(baseline,accelerations,decelerations,n)
assert(isnumeric(baseline) && numel(baseline)==n && all(isfinite(baseline(:))), 'T07C:InvalidBaseline', 'T07C check failed: InvalidBaseline');
for data={accelerations,decelerations}
    a=data{1};
    assert(isnumeric(a) && size(a,1)==2 && all(isfinite(a(:))), 'T07C:InvalidEventShape', 'T07C check failed: InvalidEventShape');
    assert(all(a(1,:)>=0) && all(a(2,:)>a(1,:)) && all(a(2,:)<=n/4+0.25), 'T07C:InvalidEventBoundary', 'T07C check failed: InvalidEventBoundary');
end
end

function assert_valid_mat(path,n)
s=load(path,'baseline','accelerations','decelerations','n_stage1');
assert(isfield(s,'n_stage1'),'T07C:OutputMissingLength', 'T07C check failed: OutputMissingLength');
if isempty(n), n=double(s.n_stage1);end
assert(double(s.n_stage1)==n,'T07C:ExistingOutputLengthMismatch', 'T07C check failed: ExistingOutputLengthMismatch');
assert_pred(s.baseline,s.accelerations,s.decelerations,n);
end

function safe_write_text(path,txt)
assert(exist(path,'file')~=2,'T07C:RefuseOverwrite', 'T07C check failed: RefuseOverwrite');
fid=fopen(path,'w');
assert(fid>=0,'T07C:CannotWriteFile', 'T07C check failed: CannotWriteFile');
cleanup=onCleanup(@()fclose(fid));
fwrite(fid,txt,'char');
end

function safe_write_error(path,txt)
% Existing error artifacts are immutable; keep the earliest failure evidence.
if exist(path,'file')==2
    fprintf(2,'T07C_PRIOR_ERROR_PRESERVED %s\n',path);
    return;
end
safe_write_text(path,txt);
end
