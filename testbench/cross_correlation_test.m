% Multichannel plotting.
% Currently only works with 7 outputs (0 to 6)
function [max_corss_corr, C, lag] = cross_correlation_test(signal1, signal2, Fs)
    signal1 = signal1 - mean(signal1);
    signal2 = signal2 - mean(signal2);

    %signal1 = alignsignals(signal1,signal2);

    [C,lag] = xcorr(signal1,signal2,'coeff');
    
    [~,I] = max(abs(C));
    SampleDiff = lag(I);
    timeDiff = SampleDiff/Fs;

    max_corss_corr = C(I);
end