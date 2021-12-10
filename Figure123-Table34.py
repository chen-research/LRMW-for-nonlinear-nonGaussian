#You probably need an nvidia gpu to run this code using numba
#This code produces Figure 1 & 2, Table 3 & 4.
#Due to randomness, the results obtained by running this code may be a little bit different (not far) from the results in the paper.

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy.special import gamma
from scipy.ndimage.interpolation import shift
from pre_defined_functions import *
from numba import jit
figure_path = "D:/Learning/Academic_Research/Projects/Estimating the memory parameter of time series/Submission/Inverse Problems/Referee Reports and Revision/Replies to Reviewer Comments/Figures/"

@jit
def compute_PjN(N,L_j):
    portion = np.ones((N-L_j+1, N-L_j+1))
    for t1 in range(L_j-1,N):
        for t2 in range(L_j-1,N):
            portion[t1-(L_j-1),t2-(L_j-1)] = max(0, L_j-1-np.abs(t1-t2))/L_j
    P_jN = np.mean(portion)
    return P_jN


@jit
def compute_QjN(N,L_j):
    portion = np.ones((N-L_j+1, N-L_j+1))
    for t1 in range(L_j-1,N):
        for t2 in range(L_j-1,N):
            portion[t1-(L_j-1),t2-(L_j-1)] =(         (L_j-1-np.abs(t1-t2)) >0         )
    Q_jN = np.mean(portion)
    return Q_jN



################
#Figure 1 
#This block plots P_{j,N} and Q_{j,N}
################
js = np.arange(3, 10)
Ns = np.array([2000,5000,8000,11000])
Lj = (np.power(2,js)-1)*3+1
P_jNs = np.zeros((len(Ns),len(js)))
Q_jNs = np.zeros((len(Ns),len(js)))

#Copute P_{j,N} and Q_{j,N}
for n in range(len(Ns)):
    for k in range(len(Lj)):
        N = Ns[n]
        lj = Lj[k]
        P_jNs[n,k] = compute_PjN(N,lj)
        Q_jNs[n,k] = compute_QjN(N,lj)

#Plot P_{j,N}
f, ax =plt.subplots(2,2,figsize=(16,8)) 
for n in range(len(Ns)):
    i = n%2
    j = int(n/2)
    #Compute the fitted line intercept and slope
    x = np.append(np.ones([len(js),1]), np.reshape(js, [-1,1]), axis=1)
    res = linear_regr(P_jNs[n],x)
    beta = res['beta']
    ax[i,j].scatter(js,P_jNs[n],color='b',label=r"$P_{j,N}, N=$"+str(Ns[n]))
    ax[i,j].plot(js,beta[0]+beta[1]*js,color='r',label=r"fitted line, slope:"+str(beta[1].round(3)))
    ax[i,j].tick_params(labelsize=14)
    ax[i,j].set_ylim([-0.1,1.1])
    ax[i,j].set_xlim([-1,9.5])
    ax[i,j].legend(loc='upper left',fontsize=15)
    ax[i,j].set_xlabel('j',fontsize=12.5)
plt.savefig(figure_path+'PjN.eps',bbox_inches="tight")
plt.show()

#Plot Q_{j,N}
f, ax =plt.subplots(2,2,figsize=(16,8)) 
for n in range(len(Ns)):
    i = n%2
    j = int(n/2)
    #Compute the fitted line intercept and slope
    x = np.append(np.ones([len(js),1]), np.reshape(js, [-1,1]), axis=1)
    res = linear_regr(Q_jNs[n],x)
    beta = res['beta']
    ax[i,j].scatter(js,Q_jNs[n],color='b',label=r"$Q_{j,N}, N=$"+str(Ns[n]))
    ax[i,j].plot(js,beta[0]+beta[1]*js,color='r',label=r"fitted line, slope:"+str(beta[1].round(3)))
    ax[i,j].tick_params(labelsize=14)
    ax[i,j].set_ylim([-0.1,1.1])
    ax[i,j].set_xlim([-1,9.5])
    ax[i,j].legend(loc='upper left',fontsize=15)
    ax[i,j].set_xlabel('j',fontsize=12.5)
plt.savefig(figure_path+'QjN.eps',bbox_inches="tight")
plt.show()
print("P_jN", P_jNs)
print("Q_jN", Q_jNs)



############
#The OLS Estimator for alpha using wavelets
############
def sigmaj_hat_squared(data, filter_name='db1', J=None, j0=1):
    """
    This function outputs the hat{sigma}_j^2 in Eq.(8) in the paper, which equals
    the average of { squared \tidle{W}_{j,t} } over t.
    
    Inputs
    ----------
    data:1darray, the data whose power law coefficient alpha to be estimated.
    filter_name:string, the name of the basic wavelet filter used.
    J:int, the highest scale involved in the log-regression. If None, it will be assigned as the smallest 
           positive integer such that N > L_J.
    
    Outputs
    ----------
    SigmaHat:1darray, size J-(j0-1), hat{sigma}_j^2 where j=j0, j0+1, ..., J.
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
    return SigmaHat
    

    
    
    
def compute_var_log2var(true_alpha, data_len, filter_name='db1', 
                num=5, low_scale=5, high_scale=10, generator='linear', FARIMA_epsilon_SD=5):
    """
    This function computes 
    *the sample variance -- Var(sigmahat_j^2/sigma_j^2), 
    *the sample mean -- E[epsilon_j], j=low_scale to high_scale,
    *the sample variance -- Var(gph_ratio) 
    *the sample mean E[u] where u = log2(gph_ratio).
    
    Inputs.
    ---------
    true_alpha:float, the alpha used to generate the simulated color noises (process whose psd follows a power law).
    data_len:int, the length of the simulated color noise.
    filter_name:string, the name of the basic wavelet filter.
    num:int, the number of colored noises simulated.
    low_scale:int, the lowest scale involved in the estimator.
    high_scale:int, the highest scale involved in the estimator.
                    For Fourier-based estimator the frequency range is (2**(-high_scale), 2**(-low_scale))
    generator:str, if equals 'linear', the linear 1/f process will be generated;
                   if equals 'LMSV', the log squared LMSV process will be generated;
    
    Outputs.
    ---------
    A list.
    lrw_ratio_mean: np.1darray, mean(sigmahat_j^2/sigma_j^2), j in [low_scale, high_scale].
    lrw_ratio_var: np.1darray, Var(sigmahat_j^2/sigma_j^2), j in [low_scale, high_scale].
    lrw_log_ratio_mean: np.1darray, mean(epsilon_j), j in [low_scale, high_scale], epsilon_j = log2(sigmahat_j^2/sigma_j^2))
    gph_f: np.1darray, the frequencies corresponding to gph_ratio_mean
    gph_ratio_mean: np.1darray,  mean of gph_ratio = (2*np.pi)*gph_Sx/(5*psd)
    gph_ratio_var: np.1darray,  Var(gph_ratio)
    gph_log_ratio_mean np.1darray,   mean of u = 2*pi*gph_psd/psd
    """
    #Generate the process
    if generator=='linear':
        X = FARIMA_gen(alpha = true_alpha, Q_d=0.1, beta_length=1000, length=data_len, num=num)
        #The formula for sigma_j^2 is Eq.(17) in the paper, for an FARIMA process c=sigma^2_epsilon
        c = np.power(2*np.pi, -true_alpha)*0.1*0.1  #Q_d is the sd of epsilon_t used to generate FARIMA, Eq.(22)
        sigma_squared = 2*c*(1-np.power(2,true_alpha-1))/(1-true_alpha) * np.power(2, (true_alpha-1)*np.arange(low_scale, high_scale+1) )
        
    elif generator=='LMSV':
        X = LMSV_gen(alpha = true_alpha, sigma=0.8, beta_length=1000, length=data_len, num=num, 
                     epsilon_sd = FARIMA_epsilon_SD)
        c = np.power(2*np.pi, -true_alpha)*FARIMA_epsilon_SD*FARIMA_epsilon_SD  #See Section 4.2
        sigma_squared = 2*c*(1-np.power(2,true_alpha-1))/(1-true_alpha) * np.power(2, (true_alpha-1)*np.arange(low_scale, high_scale+1) )
    
    #Define the storage holder for ratio=sigmahat_j^2/sigma_j^2
    lrw_ratio = np.zeros((num, high_scale-low_scale+1))
    gph_ratio = []
    for i in range(num):
        x = X[i]
        #lrw
        sigmahat2 = sigmaj_hat_squared(x, filter_name, J=high_scale, j0=low_scale)
        lrw_ratio[i] = sigmahat2/sigma_squared
        
        #gph. 
        gph_f, gph_Sx = GPH_ps(x, flb=2**(-high_scale-1), fub=2**(-low_scale), tau=2, p=5)
        if generator=='linear':
            psd = 0.1*0.1*np.power(np.abs(2*np.sin(np.pi*gph_f)), -true_alpha)   #Eq.(22)
        if generator=='LMSV':  #See Section 4.2 for the psd
            psd = FARIMA_epsilon_SD*FARIMA_epsilon_SD*np.power(np.abs(2*np.sin(np.pi*gph_f)), -true_alpha)+(np.pi**2)/2
        gph_ratio.append( (2*np.pi)*gph_Sx/(5*psd) ) #Note that GPH_ps outputs p*estimate of S(f) (following Eq (35) in Fay et al. (2009))
        
        
    #means of sigmahat_j^2/sigma_j^2, j=low_scale, low_scale+1, ..., high_scale 
    lrw_ratio_mean = np.mean(lrw_ratio, axis=0)
    lrw_ratio_var = np.std(lrw_ratio, axis=0)**2  #Var of sigmahat_j^2/sigma_j^2
    lrw_log_ratio = np.log2(lrw_ratio)            #epsilon = log2(sigmahat_j^2/sigma_j^2)
    lrw_log_ratio_mean = np.mean(lrw_log_ratio,axis=0)  #mean(epsilon)
    
    gph_ratio = np.array(gph_ratio)                #gph_ratio= (2*np.pi)*gph_Sx/(5*psd)
    gph_ratio_mean = np.mean(gph_ratio, axis=0)
    gph_ratio_var = np.std(gph_ratio, axis=0)**2
    
    gph_log_ratio = np.log2(gph_ratio)
    gph_log_ratio_mean = np.mean(gph_log_ratio,axis=0)
    
    return [lrw_ratio_mean, lrw_ratio_var, lrw_log_ratio_mean,
            gph_f, gph_ratio_mean, gph_ratio_var, gph_log_ratio_mean
           ]

def plot_4N(estimates, x_val, Ns=[2000,5000,8000,11000], x_label=None, 
            title=None, label_n=r"$P_{j,N}, N=$", legend_loc='upper left', 
            savefig_path=None, savefig_name=None, for_gph=False):
    """
    Compute the gradient of the fitted line and plot the fitted lines.
    
    Inputs.
    ------------
    estimates:np.ndarray, should be of size [len(NS), J]
    x_val:np.ndarray, the x-axis values x_val[n,:] corresponding to estimates[n,:]
    Ns:list, Ns[n] is the case for estimates[n,:].
    """
    f, ax =plt.subplots(2,2,figsize=(16,8)) 
    if title:
        plt.suptitle(title, size=20)
    for n in range(len(Ns)):
        i = n%2
        j = int(n/2)
        #Compute the fitted line intercept and slope
        if for_gph:
            regressor = -np.log(np.abs(    1-np.exp(1j*2*np.pi*x_val[n])    ))  #-ln| 1-exp(i2pi f) |
            x = np.append(np.ones([len(regressor),1]), np.reshape(regressor, [-1,1]), axis=1) #add the column of 1s 
        else:
            x = np.append(np.ones([len(x_val[n]),1]), np.reshape(x_val[n], [-1,1]), axis=1)
        
        res = linear_regr(estimates[n],x)
        beta = res['beta']
        ax[i,j].scatter(x[:,1],estimates[n],color='b',label=label_n+str(Ns[n]),s=15)
        ax[i,j].plot(x[:,1],beta[0]+beta[1]*x[:,1],color='r',label=r"fitted line, slope:"+str(beta[1].round(3)))
        ax[i,j].tick_params(labelsize=14)
        
        #set ylim
        flat_estimates = estimates[0].copy()
        for l in range(1,len(estimates)):
            flat_estimates = np.append(flat_estimates, estimates[l], axis=0)
        
        diff = np.max(flat_estimates) - np.min(flat_estimates)
        ymin = np.min(flat_estimates)-diff*0.1 
        ymax = np.max(flat_estimates)+diff*0.1 
        ax[i,j].set_ylim([ymin,ymax])
        
        #set xlim
        diff = np.max(x[:,1]) - np.min(x[:,1])
        ax[i,j].set_xlim([np.min(x[:,1])-diff*0.5, np.max(x[:,1])+diff*0.1])
        
        ax[i,j].legend(loc=legend_loc,fontsize=15)
        if isinstance(x_label,str) & ((n==1)|(n==3)):
            ax[i,j].set_xlabel(x_label,fontsize=12.5)
    if savefig_path:
        plt.savefig(savefig_path+savefig_name,bbox_inches="tight")
    plt.show()



###############
#Table 3 & 4, Figure 2 & 3 (more figures other than Figure 2 & 3 will be output)
#Note: The slopes reported in Table 3 & 4 are from legends of the figures produced by the following codes.
###############
lm_types = ['linear', 'LMSV'] #'linear', LMSV', 'NLMA'
signal_lens = np.array([3000, 6000, 9000, 12000])

true_alphas = [0.2, 0.5, 0.8]
low_scale = 3
high_scale = 9
num = 1000
filter_name='db2'

type_num = len(lm_types)
alpha_num = len(true_alphas)
length_num = len(signal_lens)


for m in range(type_num):
    for n in range(alpha_num):
        lm_type = lm_types[m]
        true_alpha = true_alphas[n]
        
        #LRMW
        LRW_ratio_means = np.zeros((length_num,high_scale-low_scale+1))
        LRW_ratio_vars = np.zeros((length_num,high_scale-low_scale+1))
        LRW_log_ratio_means = np.zeros((length_num,high_scale-low_scale+1))
        
        #GPH
        gph_freqs = []
        gph_ratio_means = []
        gph_ratio_vars = []
        gph_log_ratio_means = []
        
        for i in range(length_num):
            #LRMW
            sig_len = signal_lens[i]
            res = compute_var_log2var(true_alpha, sig_len, filter_name, num, 
                                      low_scale, high_scale, generator=lm_type)
            LRW_ratio_means[i] = res[0]  #mean of 1000 sigma_hat2/sigma_j2
            LRW_ratio_vars[i] = res[1]   #var of 1000 sigma_hat2/sigma_j2
            LRW_log_ratio_means[i] = res[2]  #mean of 1000 log(sigma_hat2/sigma_j2)
    
            #GPH
            gph_freqs.append(res[3]) 
            gph_ratio_means.append(res[4])
            gph_ratio_vars.append(res[5])
            gph_log_ratio_means.append(res[6])
        
        #Plots for LRMW
        x_val_matrix = np.array([range(low_scale,high_scale+1)]*4)
        plot_4N(estimates=LRW_ratio_means, x_val=x_val_matrix, 
                Ns=signal_lens-1000, x_label='j', 
                title='The '+lm_type+' case and true alpha = '+str(true_alpha), 
                label_n=r"$\hat{\mathbb{E}}[\hat{\sigma}^2_{j,N}/\sigma_j^2], N=$", legend_loc='lower left', 
                savefig_path=figure_path, savefig_name='LRMW-ratio-mean-'+lm_type+str(true_alpha).replace('.','')+'.eps')
        
        plot_4N(estimates=LRW_ratio_vars, x_val=x_val_matrix, 
                Ns=signal_lens-1000, x_label='j', 
                title='The '+lm_type+' case and true alpha = '+str(true_alpha), 
                label_n=r"$\hat{Var}\left(\hat{\sigma}^2_{j,N}/\sigma_j^2\right), N=$", legend_loc='upper left', 
                savefig_path=figure_path, savefig_name='LRMW-ratio-var-'+lm_type+str(true_alpha).replace('.','')+'.eps')
        
        plot_4N(estimates=LRW_log_ratio_means, x_val=x_val_matrix, 
                Ns=signal_lens-1000, x_label='j', 
                title='The '+lm_type+' case and true alpha = '+str(true_alpha), 
                label_n=r"$\hat{\mathbb{E}}[\varepsilon_{j,N}], N=$", 
                #label_n=r"$\hat{\mathbb{E}}\left[\log_{2}(\hat{\sigma}^2_{j,N}/\sigma_j^2)\right], N=$", 
                legend_loc='lower left', 
                savefig_path=figure_path, savefig_name='LRMW-log-ratio-mean-'+lm_type+str(true_alpha).replace('.','')+'.eps')
        
        #Plots for gph
        plot_4N(estimates=gph_ratio_means, x_val=gph_freqs, for_gph=True,
                Ns=signal_lens-1000, x_label=r"$g(\tilde{f}_k)$", 
                title='The '+lm_type+' case and true alpha = '+str(true_alpha), 
                label_n=r"$\hat{\mathbb{E}}[\bar{I}_{p,\tau}(\tilde{f}_k)/S_x(\tilde{f}_k)], N=$", legend_loc='lower left', 
                savefig_path=figure_path, savefig_name='gph-ratio-mean-'+lm_type+str(true_alpha).replace('.','')+'.eps')
        
        plot_4N(estimates=gph_ratio_vars, x_val=gph_freqs, for_gph=True,
                Ns=signal_lens-1000, x_label=r"$g(\tilde{f}_k)$", 
                title='The '+lm_type+' case and true alpha = '+str(true_alpha), 
                label_n=r"$\hat{Var}\left(\bar{I}_{p,\tau}(\tilde{f}_k)/S_x(\tilde{f}_k)\right), N=$", legend_loc='upper left', 
                savefig_path=figure_path, savefig_name='gph-ratio-var-'+lm_type+str(true_alpha).replace('.','')+'.eps')
        
        plot_4N(estimates=gph_log_ratio_means, x_val=gph_freqs, for_gph=True,
                Ns=signal_lens-1000, x_label=r"$g(\tilde{f}_k)$", 
                title='The '+lm_type+' case and true alpha = '+str(true_alpha), 
                label_n=r"$\hat{\mathbb{E}}[u_{j,N}], N=$",
                legend_loc='lower left', 
                savefig_path=figure_path, savefig_name='gph-log-ratio-mean-'+lm_type+str(true_alpha).replace('.','')+'.eps')