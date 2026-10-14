clc;
clear all;
[evecs, lambdas, conv] = computeHOCGTEigenpairs('/Users/gracecalkins/Local_Documents/local_code/sttgmm/data/uranus_small_cov', ...
    [0,170;170,365;365,715], 'HOCGT', 3, 7);