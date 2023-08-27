def skill_matrix(MSLA, Psi, k_n, l_n, MModes, Rm, lon, lat, T_time):
    
    '''
    Evaluate the skillfulness of each wave in fitting the daily average AVISO SSH anomaly. 
    
    Input: 
    SSHA_vector: AVISO SSH anomaly, 
    Psi (horizontal velocity and pressure structure functions), 
    k_n (zonal wavenumber), 
    l_n (latitudional wavenumber), 
    frequency, 
    longitude, latitude and time. 
    
    Output: skill matrix， SSH anomalies as vector, longitude, latitude, time and Rossby deformation radius.
    '''
    
    import numpy as np
    from tqdm import tqdm
    from numpy import linalg as LA
    from scipy import linalg
    
    Phi0 = lat.mean() # central latitude (φ0)
    Omega = 7.27e-5 # Ω is the angular speed of the earth
    Earth_radius = 6.371e6 / 1e5 #Earth_radius # meters
    Beta = 2 * Omega * np.cos(Phi0)  / Earth_radius
    f0 = 2 * Omega * np.sin(Phi0) #1.0313e-4 # 
    
    dlon = lon - lon.mean()
    dlat = lat - lat.mean()
    
    SSHA_masked = np.ma.masked_invalid(MSLA)
    SSHA_vector = np.zeros(MSLA.size)
    time_vector = np.zeros(MSLA.size)
    Iindex, Jindex, Tindex = np.zeros(MSLA.size), np.zeros(MSLA.size), np.zeros(MSLA.size)
    
    count = 0
    for tt in range(MSLA.shape[2]):
        for jj in range(MSLA.shape[0]):  # loop over latitude
            for ii in range(MSLA.shape[1]):  # loop over longitude
                if(SSHA_masked[jj, ii, tt] != np.nan): 
                    SSHA_vector[count] = MSLA[jj, ii, tt]   # MSLA subscripts:  lat, lon, time
                    #lon_vector[count] = lon[jj]
                    #lat_vector[count] = lat[ii]
                    time_vector[count] = T_time[tt]
                    Iindex[count], Jindex[count], Tindex[count] = int(ii), int(jj), int(tt)
                    count = count + 1

    H0 = np.zeros([len(SSHA_vector), 2]) # Number of data * Number of models
    skill = np.zeros([len(k_n), len(l_n), MModes])
    omega = np.zeros([len(k_n), len(l_n), MModes])
    
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                omega[kk, ll, mm] =  Beta * k_n[kk, mm] / (k_n[kk, mm] ** 2 + l_n[ll, mm] ** 2 + Rm[mm] ** -2) # non-dispersive wave

#with tqdm(total= len(k_n) * len(l_n)* MModes) as pbar:
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                for count in range(len(Iindex)):
                    # change lon, lat to (dlon, dlat = (lon, lat) - mean
                    # conversion to distance 
                    H0[count, 0] = Psi[0, mm] * np.cos(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(Tindex[count])]) 
                    H0[count, 1] = Psi[0, mm] * np.sin(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(Tindex[count])])       

                M = 2

                RR, PP = .1, 1

                HTH = np.matmul(H0.T, H0)

                for pp in range(M):
                    HTH[pp, pp] = HTH[pp, pp] +  RR/PP

                D = np.matmul(LA.inv(HTH), H0.T)  

                X_ = np.matmul(D, SSHA_vector)

                # calculate residual
                residual = SSHA_vector - np.matmul(H0, X_)

                # variance of residual
                # evaluate skill (1- rms_residual/rms_ssha_vector) and store the skill
                # skill value nn, ll, mm, = skill value
                skill[kk, ll, mm] = 1 - (np.mean(residual**2)) / (np.mean(SSHA_vector**2))

                #pbar.update(1)
                    
    return skill, SSHA_vector, Iindex, Jindex, Tindex


def inversion(Y, H_v, P_over_R):
    
    '''
    Solve for X given observations (Y), basis function (H_v) and signal to noise ratio (P_over_R).
    Return: X (amplitudes of Rossby waves)
    This is all in model space.
    '''
    
    import numpy as np
    from numpy import linalg as LA

    HTH = np.matmul(H_v.T, H_v)
    
    #P_diag = P_over_R.diagonal()
    
    #P_over_R_inv = np.zeros(P_over_R.shape)
    
    #np.fill_diagonal(P_over_R_inv, P_diag ** -1)
    
    HTH = HTH +  P_over_R #, P: uncertainty in model, R: uncertainty in data, actually R_over_P
    
    D = np.matmul(LA.inv(HTH), H_v.T)
    
    #eig, vec = LA.eig(HTH)
    
    amp = np.matmul(D, Y)
    
    Y_estimated = np.matmul(H_v, amp)
    
    return amp, Y_estimated

def inversion2(Y, H_v, P_over_R):
    
    '''
    Solve for X given observations (Y), basis function (H_v) and signal to noise ratio (P_over_R).
    Return: X (amplitudes of Rossby waves)
    This is all in model space.
    '''
    
    import numpy as np
    from numpy import linalg as LA

    #R_factor = 0.01 ** 2 # 1 cm noise
    
    R_factor = 0.001 ** 2 # .1 cm noise
    
    R_factor_inv = 1 / R_factor
    
    HTH = np.matmul(H_v.T, H_v) * R_factor_inv 
    
    P_diag = P_over_R.diagonal()
    
    P_over_R_inv = np.zeros(P_over_R.shape)
    
    np.fill_diagonal(P_over_R_inv, P_diag ** -1)
    
    HTH = HTH +  P_over_R_inv #, P: uncertainty in model, R: uncertainty in data, actually R_over_P
    
    D = np.matmul(LA.inv(HTH), H_v.T * R_factor_inv) 
    
    amp = np.matmul(D, Y)
    
    Y_estimated = np.matmul(H_v, amp)
    
    return amp, Y_estimated

def forecast_ssh(MSLA, amp, H_all):
    
    '''
    Make SSH predictions with the estimated Rossby wave amplutudes.
    Input: timestamp, estimated amplitudes, True AVISO SSH anomalies and H matrix (basis functions).
    '''
    
    import numpy as np
    from tqdm import tqdm
    from numpy import linalg as LA
    from scipy import linalg
    
    # forecast SSH
    SSHA_predicted = np.matmul(H_all, amp)
    
    time_vector = np.zeros(MSLA.size)
    lon_vector, lat_vector = np.zeros(MSLA.size),np.zeros(MSLA.size)
    Iindex, Jindex, Tindex = np.zeros(MSLA.size), np.zeros(MSLA.size), np.zeros(MSLA.size)
    
    SSHA_vector = np.zeros(MSLA.size)
    
    # flatten SSH
    count = 0
    for ii in range(MSLA.shape[0]):
        for jj in range(MSLA.shape[1]):
            for tt in range(MSLA.shape[2]):
                if(MSLA[ii, jj, tt] != np.nan): 
                    SSHA_vector[count] = MSLA[ii, jj, tt]
                    count = count + 1
                    
                    
    #print(count)
    # calculate residual variance
    residual = SSHA_vector - SSHA_predicted

    # evaluate skill (1- rms_residual/rms_ssha_vector) and store the skill
    # skill value nn, ll, mm, = skill value
    #
    residual_iter = (np.mean(residual**2)) / (np.mean(SSHA_vector**2))
    
    return SSHA_predicted, SSHA_vector, residual_iter


def reverse_vector(True_MSLA, SSHA_predicted):
    
    '''
    Reverse the vectorization.
    '''
    
    import numpy as np
    
    MSLA_est = np.zeros(True_MSLA.shape)
    
    count = 0
    for ii in range(True_MSLA.shape[0]):
        for jj in range(True_MSLA.shape[1]):
            for tt in range(True_MSLA.shape[2]):
                #if(True_MSLA[ii, jj, tt] != np.nan):
                MSLA_est[ii, jj, tt] = SSHA_predicted[count]
                count += 1
                    
    return MSLA_est


def build_h_matrix(MSLA, MModes, k_n, l_n, lon, lat, T_time, Psi, Rm, day):
    
    '''
    Build H matrix or basis function for Rossby wave model.
    
    Input:
    SSHA_vector: SSH anomalies as a vector,
    Psi (horizontal velocity and pressure structure functions), 
    k_n (zonal wavenumber), 
    l_n (latitudional wavenumber), 
    frequency, 
    longitude, latitude and time. 
    
    Output: H matrix for Rossby wave model
    
    '''
    
    import numpy as np
    
    Phi0 = lat.mean() # central latitude (φ0)
    Omega = 7.27e-5 # Ω is the angular speed of the earth
    Earth_radius = 6.371e6 / 1e5 # meters
    Beta = 2 * Omega * np.cos(Phi0) / Earth_radius 
    f0 = 2 * Omega * np.sin(Phi0) 

    dlon = lon - lon.mean()
    dlat = lat - lat.mean()
    #print('lon',lon.mean(),'lat',lat.mean())
    M = len(k_n) * len(l_n)
    H_cos, H_sin = np.zeros([MSLA.size, M]), np.zeros([MSLA.size, M])
    H_all = np.zeros([MSLA.size, M * 2])
    omega = np.zeros([len(k_n), len(l_n), MModes])
    Iindex, Jindex, Tindex = np.zeros(MSLA.size), np.zeros(MSLA.size), np.zeros(MSLA.size)
    day_use = np.zeros(MSLA.size)
    SSHA_vector = np.zeros(MSLA.size)
    
    count = 0
    for tt in range(MSLA.shape[2]):
        for jj in range(MSLA.shape[0]):
            for ii in range(MSLA.shape[1]):
                SSHA_vector[count] = MSLA[jj, ii, tt]
                day_use[count]=day+tt
                Iindex[count], Jindex[count], Tindex[count] = int(ii), int(jj), int(tt)
                count = count + 1

    nn = 0 
    for kk in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                omega[kk, ll, mm] = Beta * k_n[kk, mm] / (k_n[kk, mm] ** 2 + l_n[ll, mm] ** 2 + Rm[mm] ** -2)
                for count in range(len(Iindex)):
                    H_cos[count, nn] = Psi[0, mm] * np.cos(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(day_use[count])])
                    H_sin[count, nn] = Psi[0, mm] * np.sin(k_n[kk, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] + omega[kk, ll, mm] * T_time[int(day_use[count])])
                nn += 1
                
    H_all[:, 0::2] = H_cos 
    H_all[:, 1::2] = H_sin
    
    return H_all
    


def build_swath(swath_width, x_width, day, lon, lat):
    
    '''
     Generate the x, y, t indices for multiple satellite passings over a given swath width and time period. 
    
    '''
    
    import numpy as np
    
    x_width = len(lon)
    
    # swath 1

    xswath_index0 = np.arange(0, x_width , 1) 
    yswath_index0 = np.arange(0, swath_width, 1)
    yswath_index_left = np.ma.masked_all([x_width, swath_width])
    xswath_index_left = np.ma.masked_all([x_width, swath_width])
    for yy in range(swath_width):
        xswath_index_left[:, yy] = xswath_index0
        
    for xx in range(x_width):
        yswath_index_left[xx] = yswath_index0 + xx
        
    yswath_index_left = np.ma.masked_outside(yswath_index_left, 0, len(lat) - 1)
    xswath_index_left = np.ma.masked_outside(xswath_index_left, 0, len(lon) - 1)
    y_mask_left = np.ma.getmask(yswath_index_left)
    x_mask_left  = np.ma.getmask(xswath_index_left)
    mask_left = np.ma.mask_or(y_mask_left,x_mask_left)
    xswath_index_left = np.ma.MaskedArray(xswath_index_left, mask_left)
    yswath_index_left = np.ma.MaskedArray(yswath_index_left, mask_left)
    
    # swath 2

    xswath_index1 = np.arange(len(lon) - x_width, len(lon))
    yswath_index1 = np.arange(len(lat) - swath_width, len(lat))
    yswath_index_right = np.ma.masked_all([x_width, swath_width])
    xswath_index_right = np.ma.masked_all([x_width, swath_width])

    for yy in range(swath_width):
        xswath_index_right[:, yy] = xswath_index1
        
    for xx in range(x_width):    
        yswath_index_right[xx] = yswath_index1 - xx  
        
    yswath_index_right = np.ma.masked_outside(yswath_index_right, 0, len(lat) - 1)
    xswath_index_right = np.ma.masked_outside(xswath_index_right, 0, len(lon) - 1)
    y_mask_right = np.ma.getmask(yswath_index_right)
    x_mask_right = np.ma.getmask(xswath_index_right)
    mask_right = np.ma.mask_or(y_mask_right,x_mask_right)
    xswath_index_right = np.ma.MaskedArray(xswath_index_right, mask_right)
    yswath_index_right = np.ma.MaskedArray(yswath_index_right, mask_right)

    yvalid_index = np.append(yswath_index_left.compressed().astype(int), yswath_index_right.compressed().astype(int)) 
    xvalid_index = np.append(xswath_index_left.compressed().astype(int), xswath_index_right.compressed().astype(int))
    
    tindex, xindex, yindex = [], [], []
    xindex =  np.tile(xvalid_index, len(day))
    yindex =  np.tile(yvalid_index, len(day))
    for dd in day:
        tmp = np.tile(dd, len(yvalid_index))
        tindex = np.append(tindex, tmp)
    
    return xindex, yindex, tindex, yswath_index_left, yswath_index_right, mask_left, mask_right




def make_error(days, alpha, yswath_index_left, yswath_index_right, y_mask_left, y_mask_right):
    
    '''
    This function models the time-varying error parameters in satellite swath data, including timing error, roll error, baseline dilation error, and phase error. The roll errors and baseline dilation errors are assumed to be correlated and are generated on the satellite swath. The function takes as inputs the number of days the data is repeated, the model parameters of the roll errors and baseline dilation errors, and the swath index for swath 1 and 2, as well as the swath masks.

The output of the function includes the valid data points of the timing error, roll errors, baseline dilation error, and phase error, as well as the valid coordinates as the distance from the center of the swath ("xc1_valid") and the quadratic of that distance ("xc2_valid").
    '''
    import numpy as np
    
    # timing error
    timing_err_left, timing_err_right = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    # Roll error
    roll_err_left, roll_err_right = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    # Baseline dilation error
    baseline_dilation_err_left, baseline_dilation_err_right = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    # phase error
    phase_err_left, phase_err_right = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    phase_err_left3, phase_err_right3 = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    phase_err_left4, phase_err_right4 = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    phase_err_left5, phase_err_right5 = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    phase_err_left6, phase_err_right6 = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_right.shape)
    al, ac = roll_err_left.shape
    xc = (ac-1) / 2
    
    # swath 1
    xc1_left, xc2_left = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_left.shape)
    H_neg_left, H_pos_left = np.ma.masked_all(yswath_index_left.shape), np.ma.masked_all(yswath_index_left.shape)

    for xx in np.arange(ac):
        xc1_left[:, xx] = (xx - xc)  #* .25     #.25 degree resolution
        xc2_left[:, xx] = (xx - xc)  ** 2  #* .25
        # timing error = alpha[0] * X^0 
        timing_err_left[:, xx] = alpha[0] # * xc1_left[:, xx] # alpha[0] == alpha_timing, alpha[0] * X^0  
        # roll error = alpha[1] * X^1
        roll_err_left[:, xx] = alpha[1] * xc1_left[:, xx]  # alpha[1] == alpha_roll, alpha[1] * X^1
        # baseline dialation error = alpha[2] * X^2
        baseline_dilation_err_left[:, xx] = alpha[2] * xc2_left[:, xx] #  alpha[2] == alpha_baseline, alpha[2] * X^2
        # phase error
        H_neg_left = np.heaviside(-1 * xc1_left[:, xx], 1) 
        H_pos_left = np.heaviside(xc1_left[:, xx], 1)
        phase_err_left3[:, xx] = alpha[3] * H_neg_left  
        phase_err_left4[:, xx] = alpha[4] * xc1_left[:, xx] * H_neg_left
        phase_err_left5[:, xx] = alpha[5] * H_pos_left 
        phase_err_left6[:, xx] = alpha[6] * xc1_left[:, xx] * H_pos_left
        phase_err_left[:, xx] = phase_err_left3[:, xx] + phase_err_left4[:, xx] + phase_err_left5[:, xx] + phase_err_left6[:, xx]

    # swath 2
    xc1_right, xc2_right = np.ma.masked_all(yswath_index_right.shape), np.ma.masked_all(yswath_index_right.shape)
    H_neg_right, H_pos_right = np.ma.masked_all(yswath_index_right.shape), np.ma.masked_all(yswath_index_right.shape)

    for xx in np.arange(ac):
        xc1_right[:, xx] = (xx - xc) #* .25 #.25 degree resolution, 1deg longitude ~ 85km * .85e5
        xc2_right[:, xx] = (xx - xc)  ** 2  # * .25  #.25 degree resolution
        # timing error = alpha[0] * X^0 #IND = -7
        timing_err_right[:, xx] = alpha[0] # * xc1_right[:, xx] # alpha[0] == alpha_timing
        # roll error = alpha[1] * X^1 #IND = -6
        roll_err_right[:, xx] = alpha[1] * xc1_right[:, xx]
        # baseline dialation error # -5
        baseline_dilation_err_right[:, xx] = alpha[2] * xc2_right[:, xx]
        # phase error = alpha[2] * X^2
        H_neg_right[:, xx] = np.heaviside(-1 * xc1_right[:, xx], 1)
        H_pos_right[:, xx] = np.heaviside(xc1_right[:, xx], 1) 
        # phase error
        phase_err_right3[:, xx] = alpha[3] * H_neg_right[:, xx] # IND =-4   
        phase_err_right4[:, xx] = alpha[4] * xc1_right[:, xx] * H_neg_right[:, xx] # IND =-3
        phase_err_right5[:, xx] = alpha[5] * H_pos_right[:, xx] # -2
        phase_err_right6[:, xx] = alpha[6] * xc1_right[:, xx] * H_pos_right[:, xx] # IND = -1
        phase_err_right[:, xx] = phase_err_right3[:, xx] + phase_err_right4[:, xx] + phase_err_right5[:, xx] + phase_err_right6[:, xx]


    roll_err_left_masked = np.ma.MaskedArray(roll_err_left, y_mask_left)
    roll_err_right_masked = np.ma.MaskedArray(roll_err_right, y_mask_right)
    timing_err_left_masked = np.ma.MaskedArray(timing_err_left, y_mask_left)
    timing_err_right_masked = np.ma.MaskedArray(timing_err_right, y_mask_right)
    baseline_dilation_err_left_masked = np.ma.MaskedArray(baseline_dilation_err_left, y_mask_left)
    baseline_dilation_err_right_masked = np.ma.MaskedArray(baseline_dilation_err_right, y_mask_right)
    phase_err_left_masked = np.ma.MaskedArray(phase_err_left, y_mask_left)
    phase_err_right_masked = np.ma.MaskedArray(phase_err_right, y_mask_right)
    xc1_left_masked = np.ma.MaskedArray(xc1_left, y_mask_left)
    xc2_left_masked = np.ma.MaskedArray(xc2_left, y_mask_left)
    xc1_right_masked = np.ma.MaskedArray(xc1_right, y_mask_right)
    xc2_right_masked = np.ma.MaskedArray(xc2_right, y_mask_right)
    
    timing_err_left_valid = timing_err_left_masked.compressed() # retrieve the valid data 
    timing_err_right_valid = timing_err_right_masked.compressed() # retrieve the valid data 
    roll_err_left_valid = roll_err_left_masked.compressed() # retrieve the valid data 
    roll_err_right_valid = roll_err_right_masked.compressed() # retrieve the valid data 
    baseline_dilation_err_left_valid = baseline_dilation_err_left_masked.compressed() # retrieve the valid data 
    baseline_dilation_err_right_valid = baseline_dilation_err_right_masked.compressed() # retrieve the valid data 
    phase_err_left_valid = phase_err_left_masked.compressed() # retrieve the valid data 
    phase_err_right_valid = phase_err_right_masked.compressed() # retrieve the valid data     
    xc1_left_valid = xc1_left_masked.compressed() # retrieve the valid data 
    xc2_left_valid = xc2_left_masked.compressed() # retrieve the valid data 
    xc1_right_valid = xc1_right_masked.compressed() # retrieve the valid data 
    xc2_right_valid = xc2_right_masked.compressed() # retrieve the valid data 
    
    # concat left and right swath
    
    timing_err_valid_index = np.append(timing_err_left_valid, timing_err_right_valid) 
    roll_err_valid_index = np.append(roll_err_left_valid, roll_err_right_valid) 
    baseline_dilation_err_index = np.append(baseline_dilation_err_left_valid, baseline_dilation_err_right_valid)
    phase_err_valid_index = np.append(phase_err_left_valid, phase_err_right_valid)
    xc1_index = np.append(xc1_left_valid, xc1_right_valid)
    xc2_index = np.append(xc2_left_valid, xc2_right_valid)
    
    # repeat errors for "days"

    roll_err_valid = np.repeat(roll_err_valid_index, len(days))
    timing_err_valid = np.repeat(timing_err_valid_index, len(days)) 
    baseline_dilation_err_valid = np.repeat(baseline_dilation_err_index, len(days)) 
    phase_err_valid = np.repeat(phase_err_valid_index, len(days)) 
    xc1_valid = np.repeat(xc1_index, len(days))
    xc2_valid = np.repeat(xc2_index, len(days))
    
    
    return timing_err_valid, roll_err_valid, baseline_dilation_err_valid, phase_err_valid, phase_err_left3, phase_err_left4, phase_err_left5, phase_err_left6, xc1_valid, xc2_valid 


def make_error_over_time(days, alpha, yswath_index_left, yswath_index_right, y_mask_left, y_mask_right):
    
    '''
    This function models the time-varying error parameters in satellite swath data, including timing error, roll error, baseline dilation error, and phase error. The roll errors and baseline dilation errors are assumed to be correlated and are generated on the satellite swath. The function takes as inputs the number of days the data is repeated, the model parameters of the roll errors and baseline dilation errors, and the swath index for swath 1 and 2, as well as the swath masks.

The output of the function includes the valid data points of the timing error, roll errors, baseline dilation error, and phase error, as well as the valid coordinates as the distance from the center of the swath ("xc1_valid") and the quadratic of that distance ("xc2_valid").
    '''
    import numpy as np
    
    Tdim, ALdim, ACdim = len(days), yswath_index_left.shape[0], yswath_index_left.shape[1] # time dimension, along-track and across-track dimesion respectively
    # timing error
    timing_err_left, timing_err_right = np.ma.masked_all([Tdim, ALdim, ACdim]), np.ma.masked_all([Tdim, ALdim, ACdim])
    # Roll error
    roll_err_left, roll_err_right = np.ma.masked_all([Tdim, ALdim, ACdim]), np.ma.masked_all([Tdim, ALdim, ACdim])
    # Baseline dilation error
    baseline_dilation_err_left, baseline_dilation_err_right = np.ma.masked_all([Tdim, ALdim, ACdim]), np.ma.masked_all([Tdim, ALdim, ACdim])
    # phase error
    phase_err_left, phase_err_right = np.ma.masked_all([Tdim, ALdim, ACdim]), np.ma.masked_all([Tdim, ALdim, ACdim])
    phase_err_left3, phase_err_right3 = np.ma.masked_all([Tdim, ALdim, ACdim]), np.ma.masked_all([Tdim, ALdim, ACdim])
    phase_err_left4, phase_err_right4 = np.ma.masked_all([Tdim, ALdim, ACdim]), np.ma.masked_all([Tdim, ALdim, ACdim])
    phase_err_left5, phase_err_right5 = np.ma.masked_all([Tdim, ALdim, ACdim]), np.ma.masked_all([Tdim, ALdim, ACdim])
    phase_err_left6, phase_err_right6 = np.ma.masked_all([Tdim, ALdim, ACdim]), np.ma.masked_all([Tdim, ALdim, ACdim])
    # al, ac = ALdim, ACdim
    xc = (ACdim - 1) / 2
    
    # swath 1
    xc1_left, xc2_left = np.ma.masked_all([ALdim, ACdim]), np.ma.masked_all([ALdim, ACdim])
    H_neg_left, H_pos_left = np.ma.masked_all([ALdim, ACdim]), np.ma.masked_all([ALdim, ACdim])

    for xx in np.arange(ACdim):
        for tt in np.arange(Tdim):
            xc1_left[:, xx] = (xx - xc)  #* .25     #.25 degree resolution
            xc2_left[:, xx] = (xx - xc)  ** 2  #* .25
            # timing error = alpha[0] * X^0 
            timing_err_left[tt, :, xx] = alpha[tt, 0,0] # * xc1_left[:, xx] # alpha[0] == alpha_timing, alpha[0] * X^0  
            # roll error = alpha[1] * X^1
            roll_err_left[tt, :, xx] = alpha[tt, 1,0] * xc1_left[:, xx]  # alpha[1] == alpha_roll, alpha[1] * X^1
            # baseline dialation error = alpha[2] * X^2
            baseline_dilation_err_left[tt, :, xx] = alpha[tt, 2,0] * xc2_left[:, xx] #  alpha[2] == alpha_baseline, alpha[2] * X^2
            # phase error
            H_neg_left = np.heaviside(-1 * xc1_left[:, xx], 1) 
            H_pos_left = np.heaviside(xc1_left[:, xx], 1)
            phase_err_left3[tt, :, xx] = alpha[tt, 3,0] * H_neg_left  
            phase_err_left4[tt, :, xx] = alpha[tt, 4,0] * xc1_left[:, xx] * H_neg_left
            phase_err_left5[tt, :, xx] = alpha[tt, 5,0] * H_pos_left 
            phase_err_left6[tt, :, xx] = alpha[tt, 6,0] * xc1_left[:, xx] * H_pos_left
            phase_err_left[tt, :, xx] = phase_err_left3[tt, :, xx] + phase_err_left4[tt, :, xx] + phase_err_left5[tt, :, xx] + phase_err_left6[tt, :, xx]

    # swath 2
    xc1_right, xc2_right = np.ma.masked_all([ALdim, ACdim]), np.ma.masked_all([ ALdim, ACdim])
    H_neg_right, H_pos_right = np.ma.masked_all([ALdim, ACdim]), np.ma.masked_all([ALdim, ACdim])

    for xx in np.arange(ACdim):
        for tt in np.arange(Tdim):
            xc1_right[:, xx] = (xx - xc) #* .25 #.25 degree resolution, 1deg longitude ~ 85km * .85e5
            xc2_right[:, xx] = (xx - xc)  ** 2  # * .25  #.25 degree resolution
            # timing error = alpha[0] * X^0 #IND = -7
            timing_err_right[tt, :, xx] = alpha[tt, 0,1] # * xc1_right[:, xx] # alpha[0] == alpha_timing
            # roll error = alpha[1] * X^1 #IND = -6
            roll_err_right[tt, :, xx] = alpha[tt, 1,1] * xc1_right[:, xx]
            # baseline dialation error # -5
            baseline_dilation_err_right[tt, :, xx] = alpha[tt, 2,1] * xc2_right[:, xx]
            # phase error = alpha[2] * X^2
            H_neg_right[:, xx] = np.heaviside(-1 * xc1_right[:, xx], 1)
            H_pos_right[:, xx] = np.heaviside(xc1_right[:, xx], 1) 
            # phase error
            phase_err_right3[tt, :, xx] = alpha[tt, 3,1] * H_neg_right[:, xx] # IND =-4   
            phase_err_right4[tt, :, xx] = alpha[tt, 4,1] * xc1_right[:, xx] * H_neg_right[:, xx] # IND =-3
            phase_err_right5[tt, :, xx] = alpha[tt, 5,1] * H_pos_right[:, xx] # -2
            phase_err_right6[tt, :, xx] = alpha[tt, 6,1] * xc1_right[:, xx] * H_pos_right[:, xx] # IND = -1
            phase_err_right[tt, :, xx] = phase_err_right3[tt, :, xx] + phase_err_right4[tt, :, xx] + phase_err_right5[tt, :, xx] + phase_err_right6[tt, :, xx]


    valid_points = y_mask_left.size + y_mask_right.size
    timing_err_valid_index = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
    roll_err_valid_index = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
    baseline_dilation_err_index = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
    phase_err_valid_index = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
    xc1_index = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
    xc2_index = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
    
    for tt in range(Tdim):
        roll_err_left_masked = np.ma.MaskedArray(roll_err_left[tt], y_mask_left)
        roll_err_right_masked = np.ma.MaskedArray(roll_err_right[tt], y_mask_right)
        timing_err_left_masked = np.ma.MaskedArray(timing_err_left[tt], y_mask_left)
        timing_err_right_masked = np.ma.MaskedArray(timing_err_right[tt], y_mask_right)
        baseline_dilation_err_left_masked = np.ma.MaskedArray(baseline_dilation_err_left[tt], y_mask_left)
        baseline_dilation_err_right_masked = np.ma.MaskedArray(baseline_dilation_err_right[tt], y_mask_right)
        phase_err_left_masked = np.ma.MaskedArray(phase_err_left[tt], y_mask_left)
        phase_err_right_masked = np.ma.MaskedArray(phase_err_right[tt], y_mask_right)
        xc1_left_masked = np.ma.MaskedArray(xc1_left, y_mask_left)
        xc2_left_masked = np.ma.MaskedArray(xc2_left, y_mask_left)
        xc1_right_masked = np.ma.MaskedArray(xc1_right, y_mask_right)
        xc2_right_masked = np.ma.MaskedArray(xc2_right, y_mask_right)

        timing_err_left_valid = timing_err_left_masked.compressed() # retrieve the valid data 
        timing_err_right_valid = timing_err_right_masked.compressed() # retrieve the valid data 
        roll_err_left_valid = roll_err_left_masked.compressed() # retrieve the valid data 
        roll_err_right_valid = roll_err_right_masked.compressed() # retrieve the valid data 
        baseline_dilation_err_left_valid = baseline_dilation_err_left_masked.compressed() # retrieve the valid data 
        baseline_dilation_err_right_valid = baseline_dilation_err_right_masked.compressed() # retrieve the valid data 
        phase_err_left_valid = phase_err_left_masked.compressed() # retrieve the valid data 
        phase_err_right_valid = phase_err_right_masked.compressed() # retrieve the valid data     
        xc1_left_valid = xc1_left_masked.compressed() # retrieve the valid data 
        xc2_left_valid = xc2_left_masked.compressed() # retrieve the valid data 
        xc1_right_valid = xc1_right_masked.compressed() # retrieve the valid data 
        xc2_right_valid = xc2_right_masked.compressed() # retrieve the valid data 
        
        if tt == 0: 
            valid_points = len(timing_err_left_masked.compressed()) + len(timing_err_right_masked.compressed())
            timing_err_valid = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
            roll_err_valid = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
            baseline_dilation_err_valid = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
            phase_err_valid = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
            xc1_valid = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
            xc2_valid = np.ma.masked_all([Tdim, valid_points]) # dimensions: time, valid data points
    
        # concat left and right swath
        #print(timing_err_left_valid.shape, timing_err_right_valid.shape)
        timing_err_valid[tt] = np.append(timing_err_left_valid, timing_err_right_valid) 
        roll_err_valid[tt] = np.append(roll_err_left_valid, roll_err_right_valid) 
        baseline_dilation_err_valid[tt] = np.append(baseline_dilation_err_left_valid, baseline_dilation_err_right_valid)
        phase_err_valid[tt] = np.append(phase_err_left_valid, phase_err_right_valid)
    
        xc1_valid[tt] = np.append(xc1_left_valid, xc1_right_valid)
        xc2_valid[tt] = np.append(xc2_left_valid, xc2_right_valid)
    
    return timing_err_valid, roll_err_valid, baseline_dilation_err_valid, phase_err_valid, xc1_valid, xc2_valid 


def calculate_errors(Tdim, Valid_points, ssh_estimated_swath, ssh, cor_err, err_estimated_swath, MSLA_swath, xvalid_index, yvalid_index, lon, lat, date_time):
    import matplotlib.pyplot as plt
    import cmocean as cmo
    import numpy as np
    
    ssh_diff = ssh_estimated_swath - ssh
    err_diff = np.sqrt(np.mean((cor_err - err_estimated_swath)**2, axis=1)) / np.sqrt(np.mean(cor_err**2, axis=1))
    ssh_diff_percent = np.sqrt(np.mean(ssh_diff**2, axis=1)) / np.sqrt(np.mean(ssh**2, axis=1))
    err_map = np.zeros([Tdim, len(lon), len(lat)])
    ssh_map = np.zeros([Tdim, len(lon), len(lat)])
    ssh_true = np.zeros([Tdim, len(lon), len(lat)])
    err_true = np.zeros([Tdim, len(lon), len(lat)])
    msla_obs = np.zeros([Tdim, len(lon), len(lat)])

    for tt in range(Tdim):
        for ii in range(Valid_points):
            err_map[tt, xvalid_index[ii], yvalid_index[ii]] = err_estimated_swath[tt, ii]
            ssh_map[tt, xvalid_index[ii], yvalid_index[ii]] = ssh_estimated_swath[tt, ii]
            err_true[tt, xvalid_index[ii], yvalid_index[ii]] = cor_err[tt, ii]
            ssh_true[tt, xvalid_index[ii], yvalid_index[ii]] = ssh[tt, ii]
            msla_obs[tt, xvalid_index[ii], yvalid_index[ii]] = MSLA_swath[tt, ii]

        err_diff1 = np.sqrt(np.mean((err_true[tt] - err_map[tt])**2)) / np.sqrt(np.mean(err_true[tt]**2))
        ssh_diff1 = np.sqrt(np.mean((ssh_true[tt] - ssh_map[tt])**2)) / np.sqrt(np.mean(ssh_true[tt]**2))

    return err_diff, ssh_diff_percent, err_map, ssh_map, err_true, ssh_true


def plot_time_series(tt, ssh, MSLA_swath, ssh_estimated_swath, ssh_diff, ssh_diff_percent, cor_err, err_estimated_swath, err_diff):
    
    import matplotlib.pyplot as plt
    import cmocean as cmo

    plt.figure(figsize=(10, 8))

    plt.subplot(311)
    plt.plot(ssh[tt], 'b-', label='True SSH')
    plt.plot(MSLA_swath[tt], 'g', label = 'True SSH + Error')
    plt.plot(ssh_estimated_swath[tt], 'r--', label='Estimated SSH')
    plt.xlabel('Data point', fontsize=14)
    plt.ylabel('SSH (m)', fontsize=14)
    plt.title('SSH estimation', fontsize=16)
    plt.legend(fontsize=12)

    plt.subplot(312)
    plt.plot(ssh_diff[tt], 'b', label='True SSH - SSH estimate')
    plt.xlabel('Data point', fontsize=14)
    plt.ylabel('SSH (m)', fontsize=14)
    plt.title('True SSH - SSH estimate, ' + str(ssh_diff_percent[tt] * 100)[:5] + '%', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.subplot(313)
    plt.plot(cor_err[tt], 'b-', label='True error')
    plt.plot(err_estimated_swath[tt], 'r--', label='Estimated error')
    plt.legend(fontsize=12)
    plt.title('True error - error estimate, ' + str(err_diff[tt] * 100)[:5] + '%', fontsize=16)

def plot_ssh_err_estimate(tt, date_time, ssh, MSLA_swath, ssh_estimated_swath, ssh_diff, ssh_diff_percent, cor_err, err_estimated_swath, err_diff):
    
    import matplotlib.pyplot as plt
    import cmocean as cmo

    plot_time_series(tt, ssh, MSLA_swath, ssh_estimated_swath, ssh_diff, ssh_diff_percent, cor_err, err_estimated_swath, err_diff)
    plt.savefig('./ssh_err_estimate/ssh_err_parameter_'+  str(date_time[tt])[:10] +'.png')
    #plt.close()

def plot_ssh_err_est_maps(tt, day0, day1, lon, lat, err_true, ssh_true, ssh_map, err_map, err_diff, ssh_diff):
    
    import matplotlib.pyplot as plt
    import cmocean as cmo
    
    fig = plt.figure(figsize = (10, 12))

    plt.subplot(321)
    plt.pcolormesh(lon, lat, err_true+ ssh_true, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('a) True error + true SSH', fontsize = 14)

    plt.subplot(323)
    plt.pcolormesh(lon, lat, ssh_map, vmin = -.2, vmax = .2, cmap = cmo.cm.balance)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('c) SSH estimate, 1-stage ' + str((1- ssh_diff[tt]) * 100)[:5] + '%', fontsize = 14)

    plt.subplot(325)
    plt.pcolormesh(lon, lat, ssh_true - ssh_map, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('e) True SSH - SSH estimate, ' + str(ssh_diff[tt] * 100)[:5] + '%', fontsize = 14)

    plt.subplot(322)
    plt.pcolormesh(lon, lat, err_true, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('b) True error', fontsize = 14)

    plt.subplot(324)
    plt.pcolormesh(lon, lat, err_map, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('d) Error estimate ' +  str((1- err_diff[tt]) * 100)[:4] + '%', fontsize = 14)

    plt.subplot(326)
    plt.pcolormesh(lon, lat, err_true - err_map, cmap = cmo.cm.balance, vmin = -.2, vmax = .2)
    plt.colorbar()
    plt.xlabel('Longitude (\N{DEGREE SIGN}W)', fontsize = 14)
    plt.ylabel('Latitude (\N{DEGREE SIGN}N)', fontsize = 14)
    plt.title('f) True Error - Error estimate, ' + str(err_diff[tt] * 100)[:4] + '%' , fontsize = 14)

    plt.tight_layout()
    plt.savefig('ssh_err_est_maps/aviso_ssh_estimate_'+ str(date_time[tt])[:10] +
                '_1step_' + str(int(day1 - day0)*5) +'day.png', dpi = 300)
    #plt.close()

def calculate_ssh_err_diff(Tdim, Valid_points, ssh_estimated_swath, ssh, err_est_1step, cor_err):

    import numpy as np

    ssh_diff = np.zeros([Tdim, Valid_points])
    err_diff = np.zeros([Tdim, Valid_points])
    ssh_diff_percent, err_diff_percent = np.zeros([Tdim]), np.zeros([Tdim])

    for tt in range(Tdim):
        ssh_diff[tt] = ssh_estimated_swath[tt * Valid_points : (tt+1) * Valid_points] - ssh[tt * Valid_points : (tt+1) * Valid_points]
        err_diff[tt] = err_est_1step[tt * Valid_points : (tt+1) * Valid_points] - cor_err[tt * Valid_points : (tt+1) * Valid_points]
        ssh_diff_percent[tt] = np.sqrt(ssh_diff[tt]**2).mean() / np.sqrt(ssh[tt * Valid_points : (tt+1) * Valid_points]**2).mean()
        err_diff_percent[tt]  = np.sqrt(np.mean(err_diff[tt]**2).mean()) / np.sqrt(cor_err[tt * Valid_points : (tt+1) * Valid_points]**2).mean()

    return ssh_diff, err_diff, ssh_diff_percent, err_diff_percent

def remap_to_2d_map(tt, Valid_points, lon, lat, ssh_estimated_swath, ssh,  err_est_1step, cor_err, xvalid_index, yvalid_index):

    import numpy as np
    
    err_map = np.zeros([len(lon), len(lat)])
    ssh_map = np.zeros([len(lon), len(lat)])
    ssh_true = np.zeros([len(lon), len(lat)])
    err_true = np.zeros([len(lon), len(lat)])
    #msla_obs = np.zeros([len(lon), len(lat)])

    for ii in range(Valid_points):
        err_map[xvalid_index[ii], yvalid_index[ii]] = err_est_1step[tt * Valid_points : (tt+1) * Valid_points][ii]
        ssh_map[xvalid_index[ii], yvalid_index[ii]]  = ssh_estimated_swath[tt * Valid_points : (tt+1) * Valid_points][ii]
        err_true[xvalid_index[ii], yvalid_index[ii]] = cor_err[tt * Valid_points : (tt+1) * Valid_points][ii]
        ssh_true[xvalid_index[ii], yvalid_index[ii]] = ssh[tt * Valid_points : (tt+1) * Valid_points][ii]
        #msla_obs[xvalid_index[ii], yvalid_index[ii]] = MSLA_swath[tt * Valid_points : (tt+1) * Valid_points][ii]

    return ssh_map, ssh_true, err_map, err_true#, msla_obs
