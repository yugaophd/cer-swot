def skill_matrix(MSLA, Psi, k_n, l_n, MModes, wavespeed, lon, lat, T_time):
    
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
    
    Phi0 = 45 # central latitude (φ0)
    Omega = 7.27e-5 # Ω is the angular speed of the earth
    Earth_radius = 6.371e6 # meters
    Beta = 2 * Omega * np.cos(Phi0) / Earth_radius
    f0 = 2 * Omega * np.sin(Phi0) #1.0313e-4 # 45 N
    g = 9.81 # gravity
    
    dlon = lon - lon.mean()
    dlat = lat - lat.mean()
    
    SSHA_masked = np.ma.masked_invalid(MSLA)
    SSHA_vector = np.zeros(MSLA.size)
    time_vector = np.zeros(MSLA.size)
    lon_vector, lat_vector = np.zeros(MSLA.size),np.zeros(MSLA.size)
    Iindex, Jindex, Tindex = np.zeros(MSLA.size), np.zeros(MSLA.size), np.zeros(MSLA.size)
    
    count = 0
    for ii in range(MSLA.shape[1]):
        for jj in range(MSLA.shape[0]):
            for tt in range(MSLA.shape[2]):
                if(SSHA_masked[ii, jj, tt] != np.nan): 
                    SSHA_vector[count] = MSLA[ii, jj, tt]
                    lon_vector[count] = lon[jj]
                    lat_vector[count] = lat[ii]
                    time_vector[count] = T_time[tt]
                    Iindex[count], Jindex[count], Tindex[count] = ii, jj, tt
                    count = count + 1

    
    H0 = np.zeros([len(SSHA_vector), 2]) # Number of data * Number of models
    skill = np.zeros([len(k_n), len(l_n), MModes])
    
    Rm = wavespeed[:MModes] / f0 # Rossby deformation radius
    
    freq_n = np.zeros([len(k_n), len(l_n), MModes])
    
    for nn in range(len(k_n)):
        for ll in range(len(l_n)):
            for mm in range(MModes):
                freq_n[nn, ll, mm] = (Beta * k_n[nn, mm]) / (k_n[nn, mm] ** 2 + l_n[ll, mm] ** 2 + Rm[mm] ** (-2))

    with tqdm(total= len(k_n) * len(l_n)* MModes) as pbar:
        for nn in range(len(k_n)):
            for ll in range(len(l_n)):
                for mm in range(MModes):
                    for count in range(len(Iindex)):
                        # change lon, lat to (dlon, dlat = (lon, lat) - mea
                        H0[count, 0] = Psi[0, mm] * np.cos(k_n[nn, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])]*  - freq_n[nn, ll, mm] * T_time[int(Tindex[count])]) #conversion to distance 
                        H0[count, 1] = Psi[0, mm] * np.sin(k_n[nn, mm] * dlon[int(Iindex[count])] + l_n[ll, mm] * dlat[int(Jindex[count])] - freq_n[nn, ll, mm] * T_time[int(Tindex[count])])           

                    M = 2

                    RR, PP = .1, 1

                    HTH = np.matmul(H0.T, H0)

                    for pp in range(M):
                        HTH[pp, pp] = HTH[pp, pp] +  RR/PP

                    D = np.matmul(LA.inv(HTH), H0.T)   

                    X_ = np.matmul(D, SSHA_vector)

                    #print(X_)
                    # calculate residual
                    residual = SSHA_vector - np.matmul(H0, X_)

                    # variance of residual
                    # evaluate skill (1- rms_residual/rms_ssha_vector) and store the skill
                    # skill value nn, ll, mm, = skill value
                    skill[nn, ll, mm] = 1 - np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(SSHA_vector**2))

                    pbar.update(1)
                    
    return skill, SSHA_vector, dlon, dlat, Iindex, Jindex, Tindex, Rm


def inversion(Y, H_v, P_over_R):
    
    '''
    Solve for X given observations (Y), basis function (H_v) and signal to noise ratio (P_over_R).
    Return: X (amplitudes of Rossby waves)
    '''
    
    import numpy as np
    from numpy import linalg as LA

    HTH = np.matmul(H_v.T, H_v)
    
    HTH = HTH +  P_over_R #, P: uncertainty in model, R: uncertainty in data
    
    D = np.matmul(LA.inv(HTH), H_v.T)
    
    eig, vec = LA.eig(HTH)
    
    amp = np.matmul(D, Y)
    
    Y_estimated = np.matmul(H_v, amp)
    
    return amp, Y_estimated


def make_ssh_predictions(timestamp, amp, MSLA, H_matrix):
    
    '''
    Make SSH predictions with the estimated Rossby wave amplutudes.
    Imput: timestamp, estimated amplitudes, True AVISO SSH anomalies and H matrix (basis functions).
    '''
    
    import numpy as np
    from tqdm import tqdm
    from numpy import linalg as LA
    from scipy import linalg
    
    time_vector = np.zeros(MSLA.size)
    lon_vector, lat_vector = np.zeros(MSLA.size),np.zeros(MSLA.size)

    Iindex, Jindex, Tindex = np.zeros(MSLA.size), np.zeros(MSLA.size), np.zeros(MSLA.size)
    SSHA_vector = np.zeros(MSLA.size)
    
    count = 0
    for ii in range(MSLA.shape[0]):
        for jj in range(MSLA.shape[1]):
            for tt in range(MSLA.shape[2]):
                if(MSLA[ii, jj, tt] != np.nan): 
                    SSHA_vector[count] = MSLA[ii, jj, tt]
                    #lon_vector[count] = lon[jj]
                    #lat_vector[count] = lat[ii]
                    #time_vector[count] = T_time[tt]
                    Iindex[count], Jindex[count], Tindex[count] = ii, jj, tt
                    count = count + 1
    
    Tindex = np.repeat(timestamp, len(SSHA_vector))

    SSHA_predicted = np.matmul(H_matrix, amp)

    # calculate residual variance
    residual_iter = SSHA_vector - SSHA_predicted

    # evaluate skill (1- rms_residual/rms_ssha_vector) and store the skill
    # skill value nn, ll, mm, = skill value
    variance_explained_iter = 1 - np.sqrt(np.mean(residual_iter**2)) / np.sqrt(np.mean(SSHA_vector**2))
    
    return SSHA_predicted, SSHA_vector, variance_explained_iter


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
                if(True_MSLA[ii, jj, tt] != np.nan):
                    MSLA_est[ii, jj, tt] = SSHA_predicted[count]
                    count += 1
    return MSLA_est


def build_swath(swath_width, x_width, day, lon, lat):
    
    '''
    Make wide swath and return (x, y, t) index.
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
    yswath_index_left = np.ma.masked_outside(yswath_index_left, 0, len(lat)-1)
    xswath_index_left = np.ma.masked_outside(xswath_index_left, 0, len(lon)-1)
    y_mask_left = np.ma.getmask(yswath_index_left)
    x_mask_left  = np.ma.getmask(xswath_index_left)
    xswath_index_left = np.ma.MaskedArray(xswath_index_left, y_mask_left)
    yswath_index_left = np.ma.MaskedArray(yswath_index_left, x_mask_left)
    
    # swath 2

    xswath_index1 = np.arange(len(lon) - x_width, len(lon))
    yswath_index1 = np.arange(len(lat) - swath_width, len(lat))
    yswath_index_right = np.ma.masked_all([x_width, swath_width])
    xswath_index_right = np.ma.masked_all([x_width, swath_width])
    for yy in range(swath_width):
        xswath_index_right[:, yy] = xswath_index1
    for xx in range(x_width):    
        yswath_index_right[xx] = yswath_index1 - xx   
    yswath_index_right = np.ma.masked_outside(yswath_index_right, 0, len(lat)-1)
    xswath_index_right = np.ma.masked_outside(xswath_index_right, 0, len(lon)-1)
    y_mask_right = np.ma.getmask(yswath_index_right)
    x_mask_right = np.ma.getmask(xswath_index_right)
    xswath_index_right = np.ma.MaskedArray(xswath_index_right, y_mask_right)
    yswath_index_right = np.ma.MaskedArray(yswath_index_right, x_mask_right)

    # multiple-day swath
    #day = 1
    yvalid_index = np.append(yswath_index_left.compressed().astype(int), yswath_index_right.compressed().astype(int)) 
    xvalid_index = np.append(xswath_index_left.compressed().astype(int), xswath_index_right.compressed().astype(int))
    #tindex = np.append(np.repeat(day, len(yvalid_index)), np.repeat(day, len(yvalid_index)))
    tindex = np.repeat(day, len(yvalid_index))
    
    return xvalid_index, yvalid_index, tindex, yswath_index_left, yswath_index_right, y_mask_left, y_mask_right




def make_error(days, alpha_roll, alpha_base, yswath_index_left, yswath_index_right, y_mask_left, y_mask_right):
    
    '''
    Make correlated errors such as roll errors and baseline dialation errors on the satellite swath.
    Inpute: days repeated, model parameter of roll errors and baseline dialation errors, swath index for swath 1&2, swath mask for 1&2.
    Output: valid data points of roll errors and baseline dialation errors, valid coordinates as a distance from the center of the swath.
    '''
    import numpy as np
    
    # Roll error
    roll_err_left = np.ma.masked_all(yswath_index_left.shape)
    roll_err_right = np.ma.masked_all(yswath_index_right.shape)

    # Baseline dilation error
    baseline_dilation_err_left = np.ma.masked_all(yswath_index_left.shape)
    baseline_dilation_err_right = np.ma.masked_all(yswath_index_right.shape)

    # swath 1

    xc1_left = np.ma.masked_all(yswath_index_left.shape)
    xc2_left = np.ma.masked_all(yswath_index_left.shape)

    al, ac = roll_err_left.shape

    for xx in np.arange(ac):
        xc1_left[:, xx] = (xx - (ac-1)/2) * .25       #.25 degree resolution
        xc2_left[:, xx] = (xx - (ac-1)/2) ** 2 * .25  #.25 degree resolution
        roll_err_left[:, xx] = alpha_roll * xc1_left[:, xx]    
        baseline_dilation_err_left[:, xx] = alpha_base * xc2_left[:, xx]

    # swath 2
    xc1_right = np.ma.masked_all(yswath_index_right.shape)
    xc2_right = np.ma.masked_all(yswath_index_right.shape)

    al, ac = roll_err_right.shape

    for xx in np.arange(ac):
        xc1_right[:, xx] = (xx - (ac-1)/2) * .25      #.25 degree resolution, 1deg longitude ~ 85km * .85e5 
        xc2_right[:, xx] = (xx - (ac-1)/2) ** 2 * .25  #.25 degree resolution
        roll_err_right[:, xx] = alpha_roll * xc1_right[:, xx]    
        baseline_dilation_err_right[:, xx] = alpha_base * xc2_right[:, xx]

    roll_err_left_masked = np.ma.MaskedArray(roll_err_left, y_mask_left)
    roll_err_right_masked = np.ma.MaskedArray(roll_err_right, y_mask_right)
    baseline_dilation_err_left_masked = np.ma.MaskedArray(baseline_dilation_err_left, y_mask_left)
    baseline_dilation_err_right_masked = np.ma.MaskedArray(baseline_dilation_err_right, y_mask_right)
    xc1_left_masked = np.ma.MaskedArray(xc1_left, y_mask_left)
    xc2_left_masked = np.ma.MaskedArray(xc2_left, y_mask_left)
    xc1_right_masked = np.ma.MaskedArray(xc1_right, y_mask_right)
    xc2_right_masked = np.ma.MaskedArray(xc2_right, y_mask_right)

    roll_err_left_valid = roll_err_left_masked.compressed() # retrieve the valid data 
    roll_err_right_valid = roll_err_right_masked.compressed() # retrieve the valid data 
    baseline_dilation_err_left_valid = baseline_dilation_err_left_masked.compressed() # retrieve the valid data 
    baseline_dilation_err_right_valid = baseline_dilation_err_right_masked.compressed() # retrieve the valid data 
    xc1_left_valid = xc1_left_masked.compressed() # retrieve the valid data 
    xc2_left_valid = xc2_left_masked.compressed() # retrieve the valid data 
    xc1_right_valid = xc1_right_masked.compressed() # retrieve the valid data 
    xc2_right_valid = xc2_right_masked.compressed() # retrieve the valid data 
    
    # repeat the roll error "days" times
    roll_err_valid_index = np.append(roll_err_left_valid, roll_err_right_valid) 
    baseline_dilation_err_index = np.append(baseline_dilation_err_left_valid, baseline_dilation_err_right_valid) 
    xc1_index = np.append(xc1_left_valid, xc1_right_valid)
    xc2_index = np.append(xc2_left_valid, xc2_right_valid)

    roll_err_valid = np.tile(roll_err_valid_index, days) 
    baseline_dilation_err_valid = np.tile(baseline_dilation_err_index, days) # repeat the baseline dilation error "days" times
    xc1_valid = np.tile(xc1_index, days)
    xc2_valid = np.tile(xc2_index, days)

    return roll_err_valid, baseline_dilation_err_valid, xc1_valid, xc2_valid 


