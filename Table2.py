"""
Run the codes in this file to replicate Table 2.

Note.
Each run may produce different (but similar to those in Table 2) results due to the 
randomness of the generated processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy.special import gamma
from scipy.ndimage.interpolation import shift
from pre_defined_functions import *


########################################--Table 2 
############### -- sigma_epsilon = 2.0
# For each fixed true alpha, a number of simulated signal is generated and whose alpha are estimated,
# and the mean and SE of the estimates for the fixed true alpha are reported.
###############
lm_type = 'LMSV' 
sigma_epsilon = 2

low_scale = 3
high_scale = 9
num = 1000
signal_lens = [3000, 6000, 9000, 12000]
true_alpha = 0.5
length_num = len(signal_lens)
filter_name='db2'
gph_means, LRMW_means = [np.zeros(length_num),np.zeros(length_num)] 
gph_SEs, LRMW_SEs = [np.zeros(length_num),np.zeros(length_num)]

for i in range(length_num):
    res = alpha_est_for_LMSV(true_alpha, sigma_epsilon, signal_lens[i], 
                             filter_name, num, 
                             low_scale, high_scale)
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






########################################--Table 2 
############### -- sigma_epsilon = 3.0
# For each fixed true alpha, a number of simulated signal is generated and whose alpha are estimated,
# and the mean and SE of the estimates for the fixed true alpha are reported.
###############
lm_type = 'LMSV' 
sigma_epsilon = 3

low_scale = 3
high_scale = 9
num = 1000
signal_lens = [3000, 6000, 9000, 12000]
true_alpha = 0.5
length_num = len(signal_lens)
filter_name='db2'
gph_means, LRMW_means = [np.zeros(length_num),np.zeros(length_num)] 
gph_SEs, LRMW_SEs = [np.zeros(length_num),np.zeros(length_num)]

for i in range(length_num):
    res = alpha_est_for_LMSV(true_alpha, sigma_epsilon, signal_lens[i], 
                             filter_name, num, 
                             low_scale, high_scale)
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



############### -- sigma_epsilon = 5.0
# For each fixed true alpha, a number of simulated signal is generated and whose alpha are estimated,
# and the mean and SE of the estimates for the fixed true alpha are reported.
###############
lm_type = 'LMSV' 
sigma_epsilon = 5

low_scale = 3
high_scale = 9
num = 1000
signal_lens = [3000, 6000, 9000, 12000]
true_alpha = 0.5
length_num = len(signal_lens)
filter_name='db2'
gph_means, LRMW_means = [np.zeros(length_num),np.zeros(length_num)] 
gph_SEs, LRMW_SEs = [np.zeros(length_num),np.zeros(length_num)]

for i in range(length_num):
    res = alpha_est_for_LMSV(true_alpha, sigma_epsilon, signal_lens[i], 
                             filter_name, num, 
                             low_scale, high_scale)
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



############### -- sigma_epsilon = 7.0
# For each fixed true alpha, a number of simulated signal is generated and whose alpha are estimated,
# and the mean and SE of the estimates for the fixed true alpha are reported.
###############
lm_type = 'LMSV' 
sigma_epsilon = 7

low_scale = 3
high_scale = 9
num = 1000
signal_lens = [3000, 6000, 9000, 12000]
true_alpha = 0.5
length_num = len(signal_lens)
filter_name='db2'
gph_means, LRMW_means = [np.zeros(length_num),np.zeros(length_num)] 
gph_SEs, LRMW_SEs = [np.zeros(length_num),np.zeros(length_num)]

for i in range(length_num):
    res = alpha_est_for_LMSV(true_alpha, sigma_epsilon, signal_lens[i], 
                             filter_name, num, 
                             low_scale, high_scale)
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