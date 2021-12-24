##########################
#This file defines the functions needed for computations in Table1.py, Table2.py, and Figure123-Table34.py
##########################

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy.special import gamma
from scipy.ndimage.interpolation import shift


def linear_regr(y,x,include_SE=False):
    '''
    Input.
    ------
    y:array of floats, the response, size [len(y),1]
    x:array of arrays, the features with the first column as constants, size [len(x), 1+number of features]
    
    Output.
    -------
    beta:array of array, [[beta0],[beta1],...]
    sigma:float, the sample standard deviation of the epsilons.
    R2:float, the R2 statistic.
    '''
    p = np.linalg.inv(   np.matmul(np.transpose(x),x)   )
    q = np.matmul(p, np.transpose(x)) 
    beta = np.matmul(q, y)
    epsilon = y-np.matmul(x,beta)
    sigma = np.std(epsilon)
    y_bar = np.mean(y)
    if np.sum( (y-y_bar)*(y-y_bar) )==0:
        print("y=ybar!")
    R2 = 1 - np.sum(epsilon*epsilon)/np.sum( (y-y_bar)*(y-y_bar) )
    SE = np.std(epsilon,ddof=x.shape[1])*np.sqrt(np.diagonal(p))
    if x.shape[1]==2:
        beta_se = sigma/(np.sqrt(len(y))*np.std(x[:,1]))
        return {'beta':beta, 'sigma':sigma, 'R2':R2, 'beta_se':beta_se, 'SE':SE}
    else:
        return {'beta':beta, 'sigma':sigma, 'R2':R2, 'SE':SE}


class modwt:
    """
    This class computes and outputs the scale-j modwt or modwt_mra, according to 
    the wavelet_method book by Percival and Walden. To deal with boundaries,
    the original data will be circularly extended.
    """
    def circ_filt(self, data, filt, neg=False):
        """
        This function outputs the result of circularly filtering data with filt.
        Input.
        ------
        data: 1darray, the data to be filtered.
        filt: 1darray, the filter.
        neg: bool, True if the indices of the entries in filt are -L+1, -L+2, ..., 0
                   Negative if the indices of the entries in filt are 0, 1, 2, ..., L-1.
                   where L is the length of filt.
    
        Output.
        -------
        res:np.1darray, the filtering output, which has the same length as data.
    
        """
        n = len(data)
        m = len(filt)
        assert (n>=m), "Error: The data is shorter than the filter, which is not allowed."
    
        if neg:
            s = np.concatenate([data,data[0:m]])
            res = np.convolve(s, filt, 'full')[m-1:m-1+n] #np.convolve outputs more entries than needed
            return res
        
        else:
            s = np.concatenate([data[-m:],data])
            res = np.convolve(s, filt, 'full')[m:m+n] #np.convolve outputs more entries than needed
            return res
    
    def jth_filt_calc(self, filt, j=2):
        """
        This function calculates the jth INTERMEDIATE(intermediate) filter in the filter cascade for creating h_j,
        the filter used for filtering the original data to get the jth modwt wavelet/scaling coefficients.
    
        Inputs.
        -------
        filt:1darray, either the modwt scaling filter or wavelet filter.
        j:int, the scale of the output.
    
        Outputs.
        -------
        jth_filt:1darray, the jth intermediate filter in the filter cascade. Whether the input filter is scaling filter 
        or wavelet filter makes a difference.
    
        """
        L = len(filt)
        jth_filt = np.zeros((2**j-1) * (L-1)+1, dtype=float)
        jth_filt[np.arange(L)*2**(j-1)] = filt
        return jth_filt

    def modwt_jth_filt(self, sca_filt, wave_filt, j=2):
        """
        This function calculates the filter used for filtering the original data to 
        get the jth modwt wavelet/scaling coefficients.
        
        Inputs.
        -------
        sca_filt:1darray, the scaling filter used for the modwt e.g. the DB(4) scaling filter.
        wave_filt:1darray, the wavelet filter used for the modwt e.g. the DB(4) wavelet filter.
        
        Outputs.
        --------
        jth_sfilt:np.1darray, the filter used for iltering the original data to 
        get the jth modwt scaling coefficients.
        jth_wfilt:np.1darray, the filter used for iltering the original data to 
        get the jth modwt wavelet coefficients.

        """
        if j==1:
            return {'jth_sca_filt':sca_filt, 'jth_wav_filt':wave_filt}
    
        jth_wfilt = self.jth_filt_calc(wave_filt, j)
        jth_sfilt = self.jth_filt_calc(sca_filt, j)
    
        for i in range(1,j):
            filtr = self.jth_filt_calc(sca_filt, j-i)
            jth_wfilt = self.circ_filt(data=jth_wfilt, filt=filtr)
            jth_sfilt = self.circ_filt(data=jth_sfilt, filt=filtr)      
    
        return {'jth_sca_filt':jth_sfilt, 'jth_wav_filt':jth_wfilt}
    
    def __init__(self, wname='db4', scale=3):
        """
        The initialize function, which computes the filters for jth_scale modwt and modwt_mra.
        
        Inputs.
        -------
        scaling_filter:1darray, the scaling filter used for the modwt e.g. the DB(4) scaling filter.
        wavelet_filter:1darray, the wavelet filter used for the modwt e.g. the DB(4) wavelet filter.
        
        Filters made ready.
        --------
        1) self.jth_sfilt: np.1darray, the filter used for filtering the original data to 
        get the jth modwt scaling coefficients.
        
        2) self.jth_wfilt: np.1darray, the filter used for filtering the original data to 
        get the jth modwt wavelet coefficients.
        
        3) self.jth_mra_dfilt: np.1darray, the filter used for iltering the jth modwt wavelet coef. to 
        get the jth modwt_mra detail. 
        
        4) self.jth_mra_sfilt: np.1darray, the filter used for iltering the jth modwt scaling coef. to 
        get the jth modwt_mra smooth. 
        
        5) self.L: int, the length of the basic filter.

        """
        wavelets = pywt.Wavelet(wname) #Get the dwt filter from pywt
        scaling_filter = np.array(wavelets.dec_lo)/np.sqrt(2)  #convert to modwt scaling filters
        wavelet_filter = np.array(wavelets.dec_hi)/np.sqrt(2)  #convert to modwt wavelet filters
        
        filts = self.modwt_jth_filt(scaling_filter, wavelet_filter, scale)
        self.jth_wfilt = filts['jth_wav_filt']
        self.jth_sfilt = filts['jth_sca_filt']
        
        self.jth_mra_dfilt = np.flip(self.jth_wfilt,axis=0)
        self.jth_mra_sfilt = np.flip(self.jth_sfilt,axis=0)
        self.L = len(scaling_filter)

    def modwt(self, data):
        """
        This function outputs the modwt scale-j wavelet and scaling coefficients.
        
        Input.
        data: np.1darray, the original data to be filtered.
        
        Output.
        wave_coef: np.1darray, the jth-scale modwt wavelet coef.
        scal_coef: np.1darray, the jth-scale modwt scaling coef.
        """
        wave_coef = self.circ_filt(data, self.jth_wfilt)
        scal_coef = self.circ_filt(data, self.jth_sfilt)
        return {'wave_coef':wave_coef, 'scal_coef':scal_coef}
    
    def modwt_mra(self, data):
        """
        This function outputs the modwt scale-j detail and smooth.
        
        Input.
        data: np.1darray, the original data to be filtered.
        
        Output.
        wave_coef: np.1darray, the jth-scale modwt wavelet coef.
        scal_coef: np.1darray, the jth-scale modwt scaling coef.
        """
        wave_coef = self.circ_filt(data, self.jth_wfilt)
        scal_coef = self.circ_filt(data, self.jth_sfilt)
        detail = self.circ_filt(wave_coef, self.jth_mra_dfilt,neg=True)
        smooth = self.circ_filt(scal_coef, self.jth_mra_sfilt,neg=True)
        return {'detail':detail, 'smooth':smooth}


############
#The OLS Estimator for alpha using wavelets
############
def alpha_wavelet(data, filter_name='db1', J=None, j0=1):
    """
    This function outputs the estimated alpha by OLS using wavelets.
    
    Inputs
    ----------
    data:1darray, the data whose power law coefficient alpha to be estimated.
    filter_name:string, the name of the basic wavelet filter used.
    J:int, the highest scale involved in the log-regression. If None, it will be assigned as the smallest 
           positive integer such that N > L_J.
    
    Outputs
    ----------
    A dictionary.
    j0:int, the smallest scale used in the regression.
    J:int, the largest scale used in the regression.
    ols_alpha:float, the estimate of the power law exponent by OLS.
    """
    
    #Initialize the modwt class obj
    FilterLen = len(pywt.Wavelet(filter_name).dec_lo)  # The length of the basic filter

    # Calculate the length of filters for each scale. If FilterLen=8, then L_1 = 8, L_2 = 22, L_3 = 50, L_4 =106, L_5 = 218, L_6 = 442, L_7 = 890, L_8 =1786...
    j = np.arange(20)
    L = (2**j-1)*(FilterLen-1)+1
    del(j)

    N = len(data)
    
    
    # Scale j0 to J will be used in the wavelet estimate, and we should have N > L_J.
    if J is None:
        for i in range(1,20):
            if N < L[i]:
                J = i-1
                break
                
    ############
    #wavelet Estimate algorithm. (OLS)
    ############

    # The estimator for Var[W_(j,t)]=E[W^2_(j,t)] will be stored in Sigmahat[j].
    # The detail coefficients used has indices ranges from L(i)+1 to N, which are not affected by the boundary conditions.
    SigmaHat = np.zeros(J-j0+1)
    for j in range(j0, J+1):
        my_modwt = modwt(filter_name, j) #Initialize modwt class for scale-j
        wt = my_modwt.modwt(data)['wave_coef'] #Get the scale-j wavelet coefficients
        SigmaHat[j-j0] = sum(  wt[L[j]:]**2  )/(N-L[j]) #The jth scale modwt coefficients are stored in wt[j,:]
                                                        #SigmaHat[i] corresponds to scale i+j0, 
                                                        #So SigmaHat[0] = Scale j0, SigmaHat[J-j0] = Scale J

    #Calculate log2(Sigmahat(j)).
    logSigmaHat = np.log2(SigmaHat)

    #OLS Estimate of alpha, use scale j = j0~J
    #The first column of regressor are 1's, for the constant term in the regression, 
    #the second column of regressor are [j0,j0+1,...,J]
    regressor = np.ones([J-j0+1,2])
    regressor[:,1] = np.arange(j0,J+1)
    ols_res = linear_regr(np.reshape(logSigmaHat,[-1,1]),regressor)
    ols_alpha = ols_res['beta'][1][0]+1
    #print('ols_alpha:',ols_alpha.round(3),'R2:',ols_res['R2'].round(3))
    
    return {'j0':j0,'J':J,'ols_alpha':ols_alpha}

def GPH_ps(x, flb=0.01, fub=0.25, tau=2, p=3):
    """
    This function outputs the GPH estimate of power spectrum of x, according to
    equation (35) in "Estimators of long-memory: Fourier versus wavelets" by Fay et al. (2009)
    
    Inputs.
    --------
    x:np.1darray, the stationary signal whose power spectrum is proportional to f^(-alpha).
    flb:float, the lower bound of the wanted frequency range.
    fub:float, the upper bound of the wanted frequency range.
    tau:int, the order of taper, the larger it is, the smaller the bias should be.
    p:int, the number of psd estimates used in the pooling. The larger it is, the smaller the variance the GPH estimate is.
    
    Outputs.
    --------
    freq:np.1darray, the vector of frequencies, lambda_tilde_k.
    gph_psd:np.1darray, the GPH estimate of the power spectrum. 
                        There is a one-to-one correspondence between the entries in freq and gph_psd.
    """   
    n = len(x)
    f = np.arange(0, int((n-1)/2))/n          #f=[0,1/n, ...]
    lmda = 2*np.pi*f                          #lmda=[0,2*pi/n,4*pi/n, ...]
    h = 1-np.exp( 2*1j*np.pi*np.arange(n+1)/n ) #h=[0,1-exp(2j*pi*1/n),...]
    D_tau = np.empty(len(lmda),dtype=complex) #D_tau corresponds to frequency of 0, 1/n, 2/n,...
    denom = np.sqrt(  2*np.pi*np.sum( np.power(np.abs(h[1:]),2*tau) ) ) #(2pi*n*a_tau)^(1/2)
    for i in range(1,len(lmda)):
        D_tau[i] = np.sum( np.power(h[1:],tau) * x * np.exp(1j*np.arange(1,n+1) * lmda[i]) )/denom

    I_tau = np.abs(D_tau)*np.abs(D_tau) #I_tau corresponds to frequency of 0, 1/n, 2/n,...
    
    ################
    #Compute I_{p,tau} in equation (35)
    ################
    I_ptau = I_tau
    for i in range(1,p):
        I_ptau = I_ptau + shift(I_tau, -i, cval=0) 
    
    #According to equation (35), (p+tau)*k<(n-1)/2
    k = np.arange(0,int((n-1)/(2*(p+tau))))

    lmda_tilde = (2*(p+tau)*(k-1)+p+tau+1)*np.pi/n   #lmda_tilde[k] = lambda_tilde_k, k=0,1,2,...
    f_tilde = lmda_tilde/(2*np.pi)                   #f_tilde[k]
    
    #I_ptau corresponds to j = 0,1,2,... (f=0/n,1/n,2/n,...).
    #f_tilde[k] = lamda_tilde_k/2pi, corresponding to I_ptau[(p+tau)*(k-1)+1]. k=1,2,...
    idx = (p+tau)*(k-1)+1
    idx[0] = 0                                       #idx[0] may be negative, so set it as 0.
    I_ptau_bar = I_ptau[idx]                         #Now, f_tilde[k] corresponds to I_ptau_bar[k], except k=0
    
    f_tilde = f_tilde[1:]                            #Remove f_tilde_0.
    I_ptau_bar = I_ptau_bar[1:]                      #Remove I_ptau_bar(lambda_tilde_0).
    
    indicator = (f_tilde>=flb) & (f_tilde<=fub)
    freq = f_tilde[indicator]
    gph_psd = I_ptau_bar[indicator]
    return [freq, gph_psd]

#Define the alpha calculator for GPH estimates
def alpha_calculator(f, Sx):
    #Calculates alpha by log-regression.
    y = np.reshape(np.log(Sx[1:]),[-1,1])
    X = np.reshape(np.log(f[1:]),[-1,1])
    N = len(y)
    ones = np.ones([N,1])
    X = np.append(ones,X,axis=1)
    res = linear_regr(y,X)
    alpha = -res['beta'][1][0]
    alpha_se = res['beta_se']
    return [alpha, res['R2'], alpha_se]

def alpha_estimators(x, fhigh, flow=2**(-10), wave_j0=2, wave_J=8, wave_name ='db1'):
    """
    This function estimates the alpha for a single 1/f process.
    """
    
    #GPH estimator
    gph_f, gph_Sx = GPH_ps(x, flb=flow, fub=fhigh, tau=2, p=3)
    gph_est = alpha_calculator(gph_f, gph_Sx)
    print("The estimated alpha of the process by GPH:", gph_est[0], "Regression R2:",gph_est[1])

    
    #wavelet estimate
    filter_name = wave_name
    wave_res = alpha_wavelet(x, wave_name, J=wave_J, j0=wave_j0)
    ols_WavAlphas = wave_res['ols_alpha']
    print("The estimated alpha by ols-wavelet (LRMW):", ols_WavAlphas)

    #The log-log plot of the estimated power spectrum with GPH estimator
    plt.plot(np.log2(gph_f), np.log2(gph_Sx))
    plt.title("(GPH)Log-log plot of the power spectrum of the generated 1/f process.",color='white')
    plt.tick_params(axis='both', colors='white')
    plt.xlabel("log(frequency)",color='white')
    plt.ylabel("log(Power Spectrum)",color='white')
    plt.show()


#The FARIMA(0,alpha/2,0) process generateor (The linear process)
def FARIMA_gen(alpha = 0.5, Q_d=0.1, beta_length=100, length=5000, num=100):
    """
    This function generates the FARIMA(0,alpha/2,0) process with {beta[k]} as the MA coeffcients of FARIMA(0,alpha/2,0).
    The power spectrum of the generated process is proportional to f^(-alpha) when f -> 0.
    Written according to equation (2.46) and (2.47) in Beran (1994).
    
    {β_k} are the coefficients in the moving average representation of a FARIMA(0,alpha/2,0) process
    β_k=gamma(k+d)/(gamma(d)*gamma(k+1))
    
    The FARIMA(0,alpha/2,0) process x_t = sum_k (β_k*epsilon_{t-k}).
    
    Inputs.
    --------
    beta_length:int, the length of beta.
    length:int, the length of the generated process.
    num:int, number of processes to be generated.
    
    Ouputs.
    -------
    x:np.2darray, each row is a generated process
    """
    #The coefficients for FARIMA(0,d,0)
    d = alpha/2    
    beta = np.zeros(beta_length)  #beta[k]=gamma(k+d)/(gamma(d)*gamma(k+1)). We do not use gamma function because gamma(1000) is too large for computation.
    for k in range(beta_length):  #beta[k]=(k+d-1)*...*d*gamma(d)/(gamma(d)*k!)
        kvec = np.arange(k)       #       =(k+d-1)*...*d/k!     
        numerator = d+kvec        #       =(k+d-1)*...*d/[k*(k-1)...*1]  
        denom = kvec+1            #       =(k+d-1)/k*(k+d-2)/(k-1)*...*d/1
        beta[k] = np.prod(numerator/denom)
    
    beta_reverse = np.flip(beta,axis=0)
    
    #x will store the FARIMA(0,alpha/2,0) process Xt.
    epsilon = np.random.normal(loc=0, scale=Q_d, size=(num,length))
    x = np.zeros((num,length))
    for t in range(1,beta_length):
        x[:,t] = np.sum(beta_reverse[-(t+1):]*epsilon[:,0:t+1],axis=1)  

    for t in range(beta_length,length):
        x[:,t] = np.sum(beta_reverse*epsilon[:,t+1-beta_length:t+1],axis=1) 
    
    return x[:,beta_length:]


#LMSV generateor
def LMSV_gen(alpha = 0.5, sigma=0.8, beta_length=100, length=5000, num=100, epsilon_sd=5):
    """
    This function generates the LMSV process with {beta[k]} as the MA coeffcients of FARIMA(0,alpha/2,0).
    The power spectrum of the generated process is proportional to f^(-alpha) when f -> 0.
    Written according to equation (17) and (18) in Teyssiere and Abry (2007).
    
    r_t = σ_t*ζ_t, ζ_t ∼ N(0, 1), (17)
    σ_t = σ*exp(Xt/2), Xt ∼ FARIMA(p, d, q), (18)
    where this implementation will make p=0, q=0, and d=alpha/2.
    
    {β_k} are the coefficients in the moving average representation of a FARIMA(0,alpha/2,0) process
    β_k=gamma(k+d)/(gamma(d)*gamma(k+1))
    
    The LMSV process is np.log(r*r).
    
    Inputs.
    --------
    beta_length:int, the length of beta.
    length:int, the length of the generated process.
    num:int, number of processes to be generated.
    epsilon_sd:float, SD of epsilon_t used to generate the FARIMA(0, alpha/2, 0) process.
    
    Ouputs.
    -------
    X:np.2darray, each row is a generated process
    """
    #The coefficients for FARIMA(0,d,0)
    d = alpha/2
    beta = np.zeros(beta_length)
    for k in range(beta_length):
        kvec = np.arange(k)
        numerator = d+kvec
        denom = kvec+1
        beta[k] = np.prod(numerator/denom)     #beta[k]=gamma(k+d)/(gamma(d)*gamma(k+1))
    
    beta_reverse = np.flip(beta,axis=0)
    
    #x will store the FARIMA(0,alpha/2,0) process Xt.
    epsilon = np.random.normal(loc=0, scale=epsilon_sd, size=(num,length))
    x = np.zeros((num,length))
    for t in range(1,beta_length):
        x[:,t] = np.sum(beta_reverse[-(t+1):]*epsilon[:,0:t+1],axis=1)  

    for t in range(beta_length,length):
        x[:,t] = np.sum(beta_reverse*epsilon[:,t+1-beta_length:t+1],axis=1) 

    
    #plt.plot(x[0:])
    #plt.show()

    xi = np.random.normal(loc=0, scale=1, size=(num,x.shape[1]))
    sigma_series = sigma*np.exp(x/2)

    r = sigma_series*xi
    X = np.log(r*r)
    X = X[:,beta_length:]
    
    return X


#The generator for Nonlinear Moving Average Process
def NLMA_gen(alpha = 0.5, rho=0.1, mu=0, beta_length=100, length=5000, num=100):
    """
    This function generates the LMSV process with {beta[k]} as the MA coeffcients of FARIMA(0,alpha/2,0).
    The power spectrum of the generated process is proportional to f^(-alpha) when f -> 0.
    Written according to equation (21) in Teyssiere and Abry (2007).
    
    r_t = μ + σ_(t−1)*ε_t, σ_(t−1) = ρ + sum_i(β_i*ε_(t−i))  (21)
    
    {β_k} are the coefficients in the moving average representation of a FARIMA(0,alpha/2,0) process
    β_k=gamma(k+d)/(gamma(d)*gamma(k+1))
    
    The NLMA process is r*r.
    
    Inputs.
    --------
    beta_length:int, the length of beta.
    length:int, the length of the generated process.
    num:int, number of processes to be generated.
    
    Ouputs.
    -------
    X:np.2darray, each row is a generated process
    """

    #The Nonlinear Moving Average Process
    d = alpha/2
    epsilon = np.random.normal(loc=0, scale=1, size=(num,length))
    #epsilon = np.random.uniform(low=-1.0, high=1.0, size=(num,length))

    #The coefficients for FARIMA(0,d,0)
    beta = np.zeros((beta_length))
    for k in range(beta_length):
        kvec = np.arange(k)
        numerator = d+kvec
        denom = kvec+1
        beta[k] = np.prod(numerator/denom)     #beta[k]=gamma(k+d)/(gamma(d)*gamma(k+1))
    
    beta_reverse = np.flip(beta,axis=0)

    r = np.zeros((num,length))
    for t in range(1,beta_length):
        sigma = rho+np.sum(beta_reverse[-(t+1):-1]*epsilon[:,0:t],axis=1)
        r[:,t] = mu+sigma*epsilon[:,t] 

    for t in range(beta_length,length):
        sigma = rho+np.sum(beta_reverse[0:-1]*epsilon[:,t-beta_length+1:t],axis=1) 
        r[:,t] = mu+sigma*epsilon[:,t]
    #print(sigma, epsilon[:,t])

    #x = x[beta_length:]
    #plt.plot(x[0:])
    #plt.show()
    r = r[:,beta_length:]
    return r*r


def alpha_est(true_alpha, data_len, filter_name='db1', 
              num=5, low_scale=5, high_scale=10, generator='linear'):
    """
    This function outputs mean and sd of the estimated alphas. These alphas are
    from a number of simulated color noises with the same alpha. 
    
    Inputs.
    ---------
    true_alpha:float, the alpha used to generate the simulated color noises.
    data_len:int, the length of the simulated color noise.
    filter_name:string, the name of the basic wavelet filter.
    num:int, the number of colored noises simulated.
    low_scale:int, the lowest scale involved in the estimator.
    high_scale:int, the highest scale involved in the estimator.
                    For Fourier-based estimator the frequency range is (2**(-high_scale), 2**(-low_scale))
    generator:str, if equals 'linear', the linear 1/f process will be generated;
                   if equals 'LARCH', the LARCH process will be generated;
                   if equals 'LMSV', the LMSV process will be generated;
                   if equals 'NLMA', the NLMA process will be generated.
    
    Outputs.
    ---------
    A dictionary.
    mt_mean:float, mean of the estimated alphas by the multi-taper method.
    pd_mean:float, mean of the estimated alphas by the periodogram method.
    ols_mean:float, mean of the estimated alphas by the ols wavelet method.
    wols_mean:float, mean of the estimated alphas by the weighted ols wavelet method.
    
    mt_SE:float, sample SD of the estimated alphas by the multi-taper method.
    pd_SE:float, similar to above.
    ols_SE:float, similar to above.
    wols_SE:float, similar to above.
    """
    #Generate the process
    if generator=='linear':
        X = FARIMA_gen(alpha = true_alpha, Q_d=0.1, beta_length=1000, length=data_len, num=num)
    elif generator=='LMSV':
        X = LMSV_gen(alpha = true_alpha, sigma=0.8, beta_length=1000, length=data_len, num=num)
    elif generator=='NLMA':
        X = NLMA_gen(alpha = true_alpha, rho=1, mu=0.5, beta_length=1000, length=data_len, num=num)

    #Define the storage holder for the estiamted alpha sequence
    ols_WavAlphas, gph_alphas = [np.zeros(num), np.zeros(num)]
    for i in range(num):
        x = X[i]
        gph_f, gph_Sx = GPH_ps(x, flb=2**(-high_scale-1), fub=2**(-low_scale), tau=2, p=5)
        gph_res = alpha_calculator(np.abs( 1-np.exp(1j*2*np.pi*gph_f) ), gph_Sx)        
        wave_res = alpha_wavelet(x, filter_name, J=high_scale, j0=low_scale)
        
        ols_WavAlphas[i] = wave_res['ols_alpha']
        gph_alphas[i] = gph_res[0]
        
    #Calculate the means of gph estimates and LRMW estimates
    ols_mean = np.mean(ols_WavAlphas)
    gph_mean = np.mean(gph_alphas)
    
    #Calculate the SE of the gph estimates and LRMW estimates
    ols_SE = np.std(ols_WavAlphas)
    gph_SE = np.std(gph_alphas)
    
    return {'gph_mean':gph_mean,'LRMW_mean':ols_mean,
            'gph_SE':gph_SE,    'LRMW_SE':ols_SE}




############# -- The following two functions are for Table 2 only
#LMSV generateor
def LMSV_gen2(alpha = 0.5, sigma=0.8, sigma_epsi=3.0, beta_length=100, length=5000, num=100):
    """
    This function generates the LMSV process with {beta[k]} as the MA coeffcients of FARIMA(0,alpha/2,0).
    The power spectrum of the generated process is proportional to f^(-alpha) when f -> 0.
    Written according to equation (17) and (18) in Teyssiere and Abry (2007).
    
    r_t = σ_t*ζ_t, ζ_t ∼ N(0, 1), (17)
    σ_t = σ*exp(Xt/2), Xt ∼ FARIMA(p, d, q), (18)
    where this implementation will make p=0, q=0, and d=alpha/2.
    
    {β_k} are the coefficients in the moving average representation of a FARIMA(0,alpha/2,0) process
    β_k=gamma(k+d)/(gamma(d)*gamma(k+1))
    
    The LMSV process is np.log(r*r).
    
    Inputs.
    --------
    beta_length:int, the length of beta.
    length:int, the length of the generated process.
    num:int, number of processes to be generated.
    
    Ouputs.
    -------
    X:np.2darray, each row is a generated process
    """
    #The coefficients for FARIMA(0,d,0)
    d = alpha/2
    beta = np.zeros(beta_length)
    for k in range(beta_length):
        kvec = np.arange(k)
        numerator = d+kvec
        denom = kvec+1
        beta[k] = np.prod(numerator/denom)     #beta[k]=gamma(k+d)/(gamma(d)*gamma(k+1))
    
    beta_reverse = np.flip(beta,axis=0)
    
    #x will store the FARIMA(0,alpha/2,0) process Xt.
    epsilon = np.random.normal(loc=0, scale=sigma_epsi, size=(num,length))
    x = np.zeros((num,length))
    for t in range(1,beta_length):
        x[:,t] = np.sum(beta_reverse[-(t+1):]*epsilon[:,0:t+1],axis=1)  

    for t in range(beta_length,length):
        x[:,t] = np.sum(beta_reverse*epsilon[:,t+1-beta_length:t+1],axis=1) 

    
    #plt.plot(x[0:])
    #plt.show()

    zeta = np.random.normal(loc=0, scale=1, size=(num,x.shape[1])) 
    sigma_series = sigma*np.exp(x/2)

    r = sigma_series*zeta
    X = np.log(r*r)
    X = X[:,beta_length:]
    
    return X

def alpha_est_for_LMSV(true_alpha, sigma_ep, data_len, filter_name='db1', 
              num=5, low_scale=5, high_scale=10):
    """
    This function outputs mean and sd of the estimated alphas. These alphas are
    from a number of simulated color noises with the same alpha. 
    
    Inputs.
    ---------
    true_alpha:float, the alpha used to generate the simulated color noises.
    data_len:int, the length of the simulated color noise.
    filter_name:string, the name of the basic wavelet filter.
    num:int, the number of colored noises simulated.
    low_scale:int, the lowest scale involved in the estimator.
    high_scale:int, the highest scale involved in the estimator.
                    For Fourier-based estimator the frequency range is (2**(-high_scale), 2**(-low_scale))
    generator:str, if equals 'linear', the linear 1/f process will be generated;
                   if equals 'LARCH', the LARCH process will be generated;
                   if equals 'LMSV', the LMSV process will be generated;
                   if equals 'NLMA', the NLMA process will be generated.
    
    Outputs.
    ---------
    A dictionary.
    mt_mean:float, mean of the estimated alphas by the multi-taper method.
    pd_mean:float, mean of the estimated alphas by the periodogram method.
    ols_mean:float, mean of the estimated alphas by the ols wavelet method.
    wols_mean:float, mean of the estimated alphas by the weighted ols wavelet method.
    
    mt_SE:float, sample SD of the estimated alphas by the multi-taper method.
    pd_SE:float, similar to above.
    ols_SE:float, similar to above.
    wols_SE:float, similar to above.
    """
    #Generate the LMSV process
    X = LMSV_gen2(alpha = true_alpha, sigma_epsi=sigma_ep, sigma=0.8, beta_length=1000, length=data_len, num=num)

    #Define the storage holder for the estiamted alpha sequence
    ols_WavAlphas, gph_alphas = [np.zeros(num), np.zeros(num)]
    for i in range(num):
        x = X[i]
        gph_f, gph_Sx = GPH_ps(x, flb=2**(-high_scale-1), fub=2**(-low_scale), tau=2, p=5)
        gph_res = alpha_calculator(np.abs( 1-np.exp(1j*2*np.pi*gph_f) ), gph_Sx)        
        wave_res = alpha_wavelet(x, filter_name, J=high_scale, j0=low_scale)
        
        ols_WavAlphas[i] = wave_res['ols_alpha']
        gph_alphas[i] = gph_res[0]
        
    #Calculate the means of gph estimates and LRMW estimates
    ols_mean = np.mean(ols_WavAlphas)
    gph_mean = np.mean(gph_alphas)
    
    #Calculate the SE of the gph estimates and LRMW estimates
    ols_SE = np.std(ols_WavAlphas)
    gph_SE = np.std(gph_alphas)
    
    return {'gph_mean':gph_mean,'LRMW_mean':ols_mean,
            'gph_SE':gph_SE,    'LRMW_SE':ols_SE}