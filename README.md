# swot-cer

This project reduces the cross-track variations of SWOT correlated errors and solving for the correlated SWOT errors as part of the assimilation.

CER101_aviso_forward_model.ipynb : make forward model Y (ssh) by fitting Rossby wave model to AVISO data.
Input: AVISO SSH anomalies at 5-day intervals at the California Current System - aviso_msla_ccs_5d.nc Stratification sample at the California Current System - stratification_sample_ccs_2015-01-06.nc

Output: rossby_wave_estimate_2016-04-01_720waves.nc: Data sample of the selected waves, amplitudes, estimated SSH anomalies and residual.

CER102_aviso_inversion_swath_no_error.ipynb: AVISO Inversion with all data and swath data, no error

CER103_aviso_inversion_swath_1vs2_step.ipynb: AVISO data assimilation with correlated error reduction with two error terms, comparing 1-step and 2-step approach.

CER103_aviso_swath_1step.ipynb: AVISO data assimilation with correlated error reduction with four error terms.

CER111: multiday data assimilation with correlated error reduction with four error terms.

CER111: multiday ensemble with correlated error reduction with four error terms.

