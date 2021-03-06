"""
Run codes in this file to replicate Table 1.

Note.
1. Each run may produce different (but similar to those in Table 1) results due to the 
randomness of the generated processes.
2. You need to manually vary the value of true_alpha to get all the results in Table 1,
e.g., set it as 0.2, 0.5, and 0.8 separately and re-run the code.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy.special import gamma
from scipy.ndimage.interpolation import shift
from pre_defined_functions import *


########################################--Table 1 (true_alpha = 0.5)
############### -- For linear processes
# For each fixed true alpha, a number of simulated signal is generated and whose alpha are estimated,
# and the mean and SE of the estimates for the fixed true alpha are reported.
true_alpha = 0.5

lm_type = 'linear' 
low_scale = 3
high_scale = 9
num = 1000
signal_lens = [3000, 6000, 9000, 12000]
length_num = len(signal_lens)
filter_name='db2'
gph_means, LRMW_means = [np.zeros(length_num),np.zeros(length_num)] 
gph_SEs, LRMW_SEs = [np.zeros(length_num),np.zeros(length_num)]

for i in range(length_num):
    res = alpha_est(true_alpha, signal_lens[i], filter_name, num, 
                    low_scale, high_scale, generator=lm_type)
    gph_means[i] = res['gph_mean']
    LRMW_means[i] = res['LRMW_mean']
    
    gph_SEs[i] = res['gph_SE']
    LRMW_SEs[i] = res['LRMW_SE']
    
    print("true alpha:",true_alpha,
          "signal_len",signal_lens[i],
          "mean of gph estimates:",gph_means[i].round(3),
          "mean of LRMW estimates:",LRMW_means[i].round(3),
          "SE of gph estimates:",gph_SEs[i].round(4), 
          "SE of LRMW estimates:",LRMW_SEs[i].round(4))
    
plt.figure()
plt.plot(signal_lens, np.abs(gph_means-true_alpha), linestyle='--', color='r',label='|gph estimate error|')
plt.plot(signal_lens, np.abs(LRMW_means-true_alpha), linestyle='-.', color='g',label='|LRMW estimate error|')
plt.xlabel("Signal Length N")
plt.ylabel("Estimation error.")
plt.title("|Estimates_mean - True alpha|")
plt.legend(loc='upper left')
plt.tick_params(colors='white')
plt.show()

plt.figure()
plt.plot(signal_lens, gph_SEs, linestyle='--', color='r',label='gph estimate SE')
plt.plot(signal_lens, LRMW_SEs, linestyle='-.', color='g',label='|LRMW_wavelet estimate SE|')
plt.xlabel("Signal Length N")
plt.ylabel("SE of estimates")
plt.title("SE of estimates")
plt.legend(loc='upper left')
plt.show()


############### -- For LMSV processes
# For each fixed true alpha, a number of simulated signal is generated and whose alpha are estimated,
# and the mean and SE of the estimates for the fixed true alpha are reported.
true_alpha = 0.5

lm_type = 'LMSV' 
low_scale = 3
high_scale = 9
num = 1000
signal_lens = [3000, 6000, 9000, 12000]
length_num = len(signal_lens)
filter_name='db2'
gph_means, LRMW_means = [np.zeros(length_num),np.zeros(length_num)] 
gph_SEs, LRMW_SEs = [np.zeros(length_num),np.zeros(length_num)]

for i in range(length_num):
    res = alpha_est(true_alpha, signal_lens[i], filter_name, num, 
                    low_scale, high_scale, generator=lm_type)
    gph_means[i] = res['gph_mean']
    LRMW_means[i] = res['LRMW_mean']
    
    gph_SEs[i] = res['gph_SE']
    LRMW_SEs[i] = res['LRMW_SE']
    
    print("true alpha:",true_alpha,
          "signal_len",signal_lens[i],
          "mean of gph estimates:",gph_means[i].round(3),
          "mean of LRMW estimates:",LRMW_means[i].round(3),
          "SE of gph estimates:",gph_SEs[i].round(4), 
          "SE of LRMW estimates:",LRMW_SEs[i].round(4))
    
plt.figure()
plt.plot(signal_lens, np.abs(gph_means-true_alpha), linestyle='--', color='r',label='|gph estimate error|')
plt.plot(signal_lens, np.abs(LRMW_means-true_alpha), linestyle='-.', color='g',label='|LRMW estimate error|')
plt.xlabel("Signal Length N")
plt.ylabel("Estimation error.")
plt.title("|Estimates_mean - True alpha|")
plt.legend(loc='upper left')
plt.tick_params(colors='white')
plt.show()

plt.figure()
plt.plot(signal_lens, gph_SEs, linestyle='--', color='r',label='gph estimate SE')
plt.plot(signal_lens, LRMW_SEs, linestyle='-.', color='g',label='|LRMW_wavelet estimate SE|')
plt.xlabel("Signal Length N")
plt.ylabel("SE of estimates")
plt.title("SE of estimates")
plt.legend(loc='upper left')
plt.show()


############### -- For NLMA processes
# For each fixed true alpha, a number of simulated signal is generated and whose alpha are estimated,
# and the mean and SE of the estimates for the fixed true alpha are reported.
###############
true_alpha = 0.5

lm_type = 'NLMA' 
low_scale = 3
high_scale = 9
num = 1000
signal_lens = [3000, 6000, 9000, 12000]
length_num = len(signal_lens)
filter_name='db2'
gph_means, LRMW_means = [np.zeros(length_num),np.zeros(length_num)] 
gph_SEs, LRMW_SEs = [np.zeros(length_num),np.zeros(length_num)]

for i in range(length_num):
    res = alpha_est(true_alpha, signal_lens[i], filter_name, num, 
                    low_scale, high_scale, generator=lm_type)
    gph_means[i] = res['gph_mean']
    LRMW_means[i] = res['LRMW_mean']
    
    gph_SEs[i] = res['gph_SE']
    LRMW_SEs[i] = res['LRMW_SE']
    
    print("true alpha:",true_alpha,
          "signal_len",signal_lens[i],
          "mean of gph estimates:",gph_means[i].round(3),
          "mean of LRMW estimates:",LRMW_means[i].round(3),
          "SE of gph estimates:",gph_SEs[i].round(4), 
          "SE of LRMW estimates:",LRMW_SEs[i].round(4))
    
plt.figure()
plt.plot(signal_lens, np.abs(gph_means-true_alpha), linestyle='--', color='r',label='|gph estimate error|')
plt.plot(signal_lens, np.abs(LRMW_means-true_alpha), linestyle='-.', color='g',label='|LRMW estimate error|')
plt.xlabel("Signal Length N")
plt.ylabel("Estimation error.")
plt.title("|Estimates_mean - True alpha|")
plt.legend(loc='upper left')
plt.tick_params(colors='white')
plt.show()

plt.figure()
plt.plot(signal_lens, gph_SEs, linestyle='--', color='r',label='gph estimate SE')
plt.plot(signal_lens, LRMW_SEs, linestyle='-.', color='g',label='|LRMW_wavelet estimate SE|')
plt.xlabel("Signal Length N")
plt.ylabel("SE of estimates")
plt.title("SE of estimates")
plt.legend(loc='upper left')
plt.show()
