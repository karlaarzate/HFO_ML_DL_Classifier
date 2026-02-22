%% ========================================================================
%  PART 1: HFO PREPROCESSING PIPELINE (200ms WINDOW)
%  Project: Automated HFO Detection with ML/DL Architecture
%  
%  Description:
%  This script implements a high-throughput preprocessing pipeline for 
%  High-Frequency Oscillation (HFO) detection in intracranial EEG. 
%  It utilizes a hybrid physics-based approach to extract 200ms signal 
%  segments (snippets) and compute a multi-dimensional feature space 
%  optimized for subsequent XGBoost classification and CNN analysis.
%
%  Key Features:
%  - Windowing: 200ms total aperture (400 samples at 2000Hz).
%  - Detection: Dual-trigger logic (Energy-based + Ground Truth Injection).
%  - Feature Engineering: Includes temporal, spectral, and morphology-based 
%    metrics (SNR, Burst-Relative energy, Gabor Correlation).
% =========================================================================

clear; clc; close all;
delete(timerfindall); 

% -------------------------------------------------------------------------
% 1. DIRECTORY CONFIGURATION (Project-Relative)
% -------------------------------------------------------------------------

% Identify the location of this script (src/matlab/)
current_path = fileparts(mfilename('fullpath'));

% Move up two levels to the Project Root (HFO-Hybrid-Detection/)
project_root = fullfile(current_path, '..', '..');

% Define paths based on your new architecture
DATA_ROOT_DIR    = fullfile(project_root, 'data', 'raw'); 
OUT_BASE_DIR     = fullfile(project_root, 'data', 'processed');

% Define subdirectories for intermediate results
OUT_SNIPPETS_DIR = fullfile(OUT_BASE_DIR, 'SNIPPETS');
OUT_FEATURES_DIR = fullfile(OUT_BASE_DIR, 'FEATURES');

% Patients to process
PATIENTS_TO_RUN  = [2, 7, 8, 12, 18, 19, 20]; 

% Create directories if they do not exist
if ~exist(OUT_SNIPPETS_DIR, 'dir'), mkdir(OUT_SNIPPETS_DIR); end
if ~exist(OUT_FEATURES_DIR, 'dir'), mkdir(OUT_FEATURES_DIR); end

% Diagnostic Log
fprintf('[INFO] Data Source: %s\n', DATA_ROOT_DIR);
fprintf('[INFO] Saving Outputs to: %s\n', OUT_BASE_DIR);

% -------------------------
% 2. SIGNAL PROCESSING HYPERPARAMETERS
% -------------------------
fs = 2000;              % Sampling frequency (Hz)
gateWinSec = 0.008;     % Integration window for energy detection
gateHopSec = 0.002;     % Window hop for detection resolution
zThr       = 4;         % Z-score threshold for Line-Length detection
mergeSec   = 0.020;     % Temporal proximity threshold for merging events
snippetHalfSec = 0.100; % 100ms radius for 200ms total extraction
snippetLen     = 400;   % Total sample length (0.2s * 2000Hz)

% -------------------------
% 3. MAIN PROCESSING PIPELINE
% -------------------------
for pIdx = 1:numel(PATIENTS_TO_RUN)
    targetPatient = PATIENTS_TO_RUN(pIdx);
    PATIENT_DIR = fullfile(DATA_ROOT_DIR, sprintf('pat%d', targetPatient));
    if ~exist(PATIENT_DIR, 'dir'), continue; end
    
    pFiles = dir(fullfile(PATIENT_DIR, '*.mat'));
    
    for f = 1:numel(pFiles)
        currentFile = pFiles(f).name;
        fullPath = fullfile(PATIENT_DIR, currentFile);
        try
            S = load(fullPath);
            if isfield(S, 'data'), data = S.data; else, data = S; end
        catch, continue; end
        
        if ~isfield(data, 'x'), continue; end
        nBip = size(data.BipChOrder,2); nSamp = size(data.x,2);
        
        % -----------------------------------------
        % LOAD CLINICAL GROUND TRUTH (GT)
        % -----------------------------------------
        R_std = []; FR_std = [];
        if isfield(data, 'R') && ~isempty(data.R)
            R_std = data.R; 
            R_std(:,2:3) = data.R(:,2:3) / fs; % Convert samples to seconds
        end
        if isfield(data, 'FR') && ~isempty(data.FR)
            FR_std = data.FR; 
            FR_std(:,2:3) = data.FR(:,2:3) / fs; 
        end
        
        % -----------------------------------------
        % SIGNAL CONDITIONING AND FILTERING
        % -----------------------------------------
        % Common average removal and 50Hz Notch filter
        x_raw = data.x - mean(data.x, 2);
        dNotch = designfilt('bandstopiir','FilterOrder',4, ...
            'HalfPowerFrequency1',49,'HalfPowerFrequency2',51,'SampleRate',fs);
        
        % Bipolar Montage derivation
        xBip = zeros(nBip, nSamp, 'single');
        for b = 1:nBip
            xBip(b,:) = filtfilt(dNotch, x_raw(data.BipChOrder(1,b),:) - x_raw(data.BipChOrder(2,b),:));
        end
        
        % HFO Bandpass Filtering (80-500 Hz)
        [bBP,aBP] = butter(4,[80 500]/(fs/2),'bandpass');
        xBP = zeros(size(xBip),'single');
        for b = 1:nBip, xBP(b,:) = filtfilt(bBP,aBP,xBip(b,:)); end
        
        % -----------------------------------------
        % CANDIDATE DETECTION AND FEATURE EXTRACTION
        % -----------------------------------------
        Snippets = {}; Meta = []; TraceMeta = {}; 
        L_R=[]; L_FR=[]; L_FRandR=[]; L_Any=[];
        RMS=[]; LL=[]; MaxA=[]; Skew=[]; Kurt=[]; P2R=[]; ZC=[]; En=[]; MA=[]; Cr=[]; 
        Va=[]; GaborCorr=[]; PowR=[]; PowFR=[]; PeakF=[]; FR_R_Ratio=[]; Dur=[]; 
        SNR=[]; BurstEnergyRatio=[]; 
        
        for b = 1:nBip
            s = xBP(b,:); s_raw = xBip(b,:);
            
            % 1. Energy-based Candidate Detection (Line-Length)
            ll_trace = movsum(abs(diff(s)), round(gateWinSec*fs), 'Endpoints', 'discard');
            ll_trace = ll_trace(1:round(gateHopSec*fs):end);
            z = (ll_trace - median(ll_trace)) / max(mad(ll_trace,1), eps);
            hits_energy = find(z > zThr);
            
            % 2. Ground Truth Injection (Ensuring inclusion of known HFOs)
            hits_GT = [];
            if ~isempty(R_std), idxR = find(R_std(:,1)==b); for i=1:numel(idxR), hits_GT(end+1) = round(((R_std(idxR(i),2)+R_std(idxR(i),3))/2)*fs/(gateHopSec*fs)); end; end
            if ~isempty(FR_std), idxFR = find(FR_std(:,1)==b); for i=1:numel(idxFR), hits_GT(end+1) = round(((FR_std(idxFR(i),2)+FR_std(idxFR(i),3))/2)*fs/(gateHopSec*fs)); end; end
            
            hits = unique(sort([hits_energy(:); hits_GT(:)]))';
            if isempty(hits), continue; end
            
            % 3. Event Merging and Temporal Clustering
            times = (hits * (gateHopSec*fs)) / fs; 
            merged_centers = times(1); merged_durations = gateWinSec; 
            for k = 2:length(times)
                diff_t = times(k) - merged_centers(end);
                if diff_t > mergeSec
                    merged_centers(end+1) = times(k); merged_durations(end+1) = gateWinSec;
                else
                    merged_durations(end) = merged_durations(end) + diff_t; 
                end
            end
            
            % 4. Multi-Domain Feature Engineering
            halfLen = round(snippetHalfSec*fs);
            for idx = 1:length(merged_centers)
                t = merged_centers(idx); t_samp = round(t * fs); 
                c0 = t_samp - halfLen; c1 = c0 + snippetLen - 1;
                if c0 < 1 || c1 > nSamp, continue; end
                
                snip_raw = s_raw(c0:c1);
                snip_norm = (snip_raw - mean(snip_raw)) / max(std(snip_raw), eps);
                
                % Signal-to-Noise Ratio (SNR) Analysis
                % Metric: Variance Ratio between the Burst Core (40ms) and Baseline
                center_idx = round(length(snip_raw)/2);
                burst_win = snip_raw(center_idx-40:center_idx+40);
                baseline_win = snip_raw(1:80);
                this_SNR = var(burst_win) / (var(baseline_win) + 1e-9);
                
                % Fast Fourier Transform (FFT) for Spectral Density
                L = length(snip_raw); Y = fft(snip_raw); P2 = abs(Y/L);
                P1 = P2(1:floor(L/2)+1); P1(2:end-1) = 2*P1(2:end-1); frq = fs*(0:(floor(L/2)))/L;
                [~,maxIdx] = max(P1); this_PeakF = frq(maxIdx);
                
                % Morphological Template Matching (Gabor Logic)
                if this_PeakF < 80, atomFreq = 80; elseif this_PeakF > 600, atomFreq = 600; else, atomFreq = this_PeakF; end
                t_vec = linspace(-snippetHalfSec, snippetHalfSec, L);
                sigma = 1 / (2*pi*(atomFreq/4)); 
                atom = exp(-t_vec.^2 / (2*sigma^2)) .* cos(2*pi*atomFreq*t_vec);
                xc = xcorr(snip_norm, atom, 'coeff');
                
                % Data Storage Aggregation
                Snippets{end+1} = snip_norm(:)'; 
                Meta(end+1,:) = [targetPatient, b, t, c0]; 
                TraceMeta{end+1,1} = currentFile; 
                
                % Ground Truth Labeling per Segment
                isR = 0; isFR = 0;
                if ~isempty(R_std), isR = any(R_std(R_std(:,1)==b, 2) <= t & R_std(R_std(:,1)==b, 3) >= t); end
                if ~isempty(FR_std), isFR = any(FR_std(FR_std(:,1)==b, 2) <= t & FR_std(FR_std(:,1)==b, 3) >= t); end
                
                L_R(end+1,1)=isR; L_FR(end+1,1)=isFR; L_FRandR(end+1,1)=(isR&&isFR); L_Any(end+1,1)=(isR||isFR);
                RMS(end+1,1)=sqrt(mean(snip_raw.^2)); LL(end+1,1)=sum(abs(diff(snip_raw))); MaxA(end+1,1)=max(abs(snip_raw));
                Skew(end+1,1)=skewness(snip_raw); Kurt(end+1,1)=kurtosis(snip_raw); 
                P2R(end+1,1)=max(abs(snip_raw))/(sqrt(mean(snip_raw.^2))+1e-9); ZC(end+1,1)=sum(abs(diff(sign(snip_raw-mean(snip_raw)))))/2;
                En(end+1,1)=sum(snip_raw.^2); MA(end+1,1)=mean(abs(snip_raw));
                Cr(end+1,1)=max(abs(snip_raw))/(mean(abs(snip_raw))+1e-9); Va(end+1,1)=var(snip_raw);
                PowR(end+1,1)=sum(P1(frq >= 80 & frq <= 250).^2); 
                PowFR(end+1,1)=sum(P1(frq > 250 & frq <= 500).^2); 
                PeakF(end+1,1)=this_PeakF; FR_R_Ratio(end+1,1)=PowFR(end)/ (PowR(end) + 1e-9);
                GaborCorr(end+1,1)=max(abs(xc));
                Dur(end+1,1) = merged_durations(idx);
                SNR(end+1,1) = this_SNR;
                BurstEnergyRatio(end+1,1) = sum(burst_win.^2) / (sum(snip_raw.^2) + 1e-9);
            end
        end
        
        % -----------------------------------------
        % EXPORT RESULTS (PARQUET & MAT)
        % -----------------------------------------
        if ~isempty(Snippets)
            T = table(Meta(:,1), Meta(:,2), Meta(:,3), Meta(:,4), string(TraceMeta), ...
                      logical(L_R), logical(L_FR), logical(L_FRandR), logical(L_Any), ...  
                      RMS, LL, MaxA, Skew, Kurt, P2R, ZC, En, MA, Cr, Va, ...
                      PowR, PowFR, PeakF, FR_R_Ratio, GaborCorr, Dur, SNR, BurstEnergyRatio, ... 
                      'VariableNames', {'Patient','Channel','TimeSec','AbsStartSample','SourceFile',...
                      'yR','yFR','yFRandR','yAnyHFO',...
                      'RMS','LineLength','MaxAmp','Skewness','Kurtosis','Peak2RMS','ZeroCross','Energy',...
                      'MeanAbs','Crest','Variance','TF_BandPow_ripple', ...
                      'TF_BandPow_fastRipple', 'TF_PeakFreq', 'TF_FR_R_Ratio', 'GaborCorrelation', ...
                      'Duration', 'SNR_Burst', 'BurstEnergyRatio'});
            
            % Save features as Parquet for optimal compatibility with Python/XGBoost
            parquetwrite(fullfile(OUT_FEATURES_DIR, sprintf('pat%d_%s_features.parquet', targetPatient, replace(currentFile,'.mat',''))), T);
            
            % Save snippets for CNN input
            SNIP.snippet = Snippets; SNIP.label_matrix = [L_R, L_FR, L_FRandR]; 
            save(fullfile(OUT_SNIPPETS_DIR, sprintf('pat%d_%s_snippets.mat', targetPatient, replace(currentFile,'.mat',''))), 'SNIP', '-v7.3');
        end
    end
end
fprintf('200ms HYBRID PREPROCESSING COMPLETE.\n');
