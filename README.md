# SWOT-CER Project Documentation

## Overview
The SWOT-CER project focuses on enhancing the accuracy of SWOT (Surface Water and Ocean Topography) data. It aims to reduce cross-track variations and effectively address correlated SWOT errors within the data assimilation process. This initiative employs a series of Jupyter notebooks, each designed to tackle different aspects of error reduction and data modeling using AVISO data.

## Notebooks and Functions

### CER101_ccs_forward_model.ipynb
- **Purpose**: Constructs the forward model Y (ssh) using the Rossby wave model with AVISO data over multiple days.
- **Input**:
  - AVISO SSH anomalies at 1-day intervals in the California Current System (`aviso_msla_ccs_5d.nc`).
  - Stratification sample in the California Current System (`stratification_sample_ccs_2015-01-06.nc`).
- **Output**:
  - NetCDF file (`./rossby_wave_estimate_[date]_[number_of_waves]waves_data.nc`) with selected waves, amplitudes, estimated SSH anomalies, and residual.
  - Produces Fig.1 - 3 in the manuscript.

### CER111
- **Function**: Multiday data assimilation emphasizing correlated error reduction with four error terms.

### CER113 and CER115
- **Function**: Multiday ensemble analysis for correlated error reduction with four error terms.

## Contribution and Usage
This project is intended for researchers and developers in oceanographic data analysis and satellite data assimilation. Contributions should focus on enhancing error reduction techniques and the precision of forward modeling in satellite oceanography.

For detailed instructions and additional information, refer to the documentation within each notebook.
