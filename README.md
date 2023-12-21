# SWOT-CER Project Documentation

## Overview
The SWOT-CER project focuses on enhancing the accuracy of SWOT (Surface Water and Ocean Topography) data. It aims to reduce cross-track variations and effectively address correlated SWOT errors within the data assimilation process. This initiative employs a series of Jupyter notebooks, each designed to tackle different aspects of error reduction and data modeling using AVISO data.

## Notebooks and Functions

### CER101_aviso_forward_model.ipynb
- **Purpose**: Develops a forward model, Y (ssh), applying the Rossby wave model to AVISO data.
- **Input**:
  - AVISO SSH anomalies at 5-day intervals in the California Current System (`aviso_msla_ccs_5d.nc`).
  - Stratification sample in the California Current System (`stratification_sample_ccs_2015-01-06.nc`).

### CER101b_aviso_forward_model-copy1.ipynb
- **Purpose**: Constructs the forward model Y (ssh) using the Rossby wave model with AVISO data over multiple days.
- **Input**:
  - AVISO SSH anomalies at 1-day intervals in the California Current System (`aviso_msla_ccs_5d.nc`).
  - Stratification sample in the California Current System (`stratification_sample_ccs_2015-01-06.nc`).
- **Output**: NetCDF file (`./rossby_wave_estimate_[date]_[number_of_waves]waves_data.nc`) with selected waves, amplitudes, estimated SSH anomalies, and residual.

### CER102_aviso_inversion_swath_no_error.ipynb
- **Function**: Performs AVISO inversion using all data and swath data, excluding error considerations.

### CER103_aviso_inversion_swath_1vs2_step.ipynb
- **Function**: Implements AVISO data assimilation focusing on correlated error reduction with two error terms, comparing 1-step and 2-step approaches.

### CER103_aviso_swath_1step.ipynb
- **Function**: AVISO data assimilation targeting correlated error reduction integrating four error terms.

### CER111
- **Function**: Multiday data assimilation emphasizing correlated error reduction with four error terms.

### CER113
- **Function**: Multiday ensemble analysis for correlated error reduction with four error terms.

## Contribution and Usage
This project is intended for researchers and developers in oceanographic data analysis and satellite data assimilation. Contributions should focus on enhancing error reduction techniques and the precision of forward modeling in satellite oceanography.

For detailed instructions and additional information, refer to the documentation within each notebook.
