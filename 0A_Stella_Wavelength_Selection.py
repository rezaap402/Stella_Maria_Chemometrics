import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression

def VIP(x, y):
    # Perform a PLS algorithm to extract the X and Y loadings and scores, and weights.
    A = 18
    plsvv = PLSRegression(n_components=A, scale=False)
    plsvv.fit(x, y)
    T = plsvv.x_scores_
    P = plsvv.x_loadings_
    Q = plsvv.y_loadings_
    W = plsvv.x_weights_

    # Now we compute the VIP scores.
    a, b = W.shape
    vips = np.zeros((a,))
    s = np.diag(T.T @ T @ Q.T @ Q).reshape(b, -1)
    stot = np.sum(s)
    vips = np.sqrt(a * (W ** 2 @ s) / stot)
    return vips

fontsize=20;

Op_file = 'E:\FROM COMPUTERS_26 DES 2025\MyPhD\Stella_data';

wavelength = np.arange(900, 2200+1)
# calibration
Cfile = pd.read_excel(Op_file+ str(r'\M10-v2_Cal70')+".xlsx")
X_train = Cfile.values[:, 1:]; # if your data is not % reflectance, just delete the *100
y_train = Cfile.values[:, 0];

# prediction/validation
Pfile = pd.read_excel(Op_file+ str(r'\M10-v2_Pred30')+".xlsx");
X_test = Pfile.values[:, 1:];
y_test = Pfile.values[:, 0];

sr,sc = np.shape(X_train);
corr_coef = np.zeros(sc)
corr_coef2 = np.zeros(sc)
p_value = np.zeros(sc)

for ib in range(sc):
    corr_coef[ib] = np.corrcoef(X_train[:,ib], y_train)[0, 1]
    corr_coef2[ib], p_value[ib] = pearsonr(X_train[:, ib], y_train)

# VIP
cutoff = 1.0
vips = VIP(X_train, y_train);

sel_idx = np.where(vips>=cutoff)[0]

# new
x_cal_vip = X_train[:, sel_idx]
x_val_vip = X_test[:, sel_idx]

# new_calibration = np.concatenate((y_train.reshape(-1,1), x_cal_vip), axis=1)
# new_calibration_2 = pd.DataFrame(new_calibration)
# new_calibration_2.to_excel(Op_file+ str(r'\M10-v2_Cal70_VIP')+".xlsx", index=False)

# new_prediction = np.concatenate((y_test.reshape(-1,1), x_val_vip), axis=1)
# new_prediction_2 = pd.DataFrame(new_prediction)
# new_prediction_2.to_excel(Op_file+ str(r'\M10-v2_Pred30_VIP')+".xlsx", index=False)

# wavelength2 = pd.DataFrame(wavelength[sel_idx])
# wavelength2.to_excel(Op_file+ str(r'\Wavelength_NIR_VIP')+".xlsx", index=False)

figure_1 = plt.figure(figsize=(12,8))
plt.plot(wavelength, corr_coef2, lw=2, c='blue', label='r')
plt.plot(wavelength, p_value, lw=2, c='orange', label='p-value')
plt.axhline(0.05)
plt.xlabel('Wavelength (nm)', fontsize=fontsize-2)
plt.ylabel('Correlation coefficient', fontsize=fontsize-2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize-3)
plt.tight_layout()

figure_2 = plt.figure(figsize=(12,8))
plt.plot(wavelength, vips, lw=2, c='blue')
plt.axhline(1, lw=0.5, linestyle='--')
plt.xlabel('Wavelength (nm)', fontsize=fontsize-2)
plt.ylabel('VIP score', fontsize=fontsize-2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()

mean_s = np.mean(X_train,axis=0)
figure_3 = plt.figure(figsize=(10,6))
plt.plot(wavelength, mean_s, color='navy', lw=2, label='Full spectra')
plt.scatter(wavelength[sel_idx], mean_s[sel_idx], color='red', marker='o', label='VIP-spectra')
plt.xlabel('Wavelength (nm)', fontsize=fontsize-2)
plt.ylabel('Absorbance', fontsize=fontsize-2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize-3)
plt.tight_layout()
# figure_3.savefig(Op_file+ str(r'\Spectra_VIP')+".svg", bbox_inches='tight')

figure_4 = plt.figure(figsize=(10,6))
plt.plot(wavelength, X_train.T, lw=2)
plt.text(990, 0.1, s='990', ha='center', va='center', fontsize=fontsize-4)
plt.text(1930, 0.55, s='1930', ha='center', va='center', fontsize=fontsize-4)
plt.text(2099, 0.48, s='2099', ha='center', va='center', fontsize=fontsize-4)
plt.ylim([0, 0.65])
plt.xlabel('Wavelength (nm)', fontsize=fontsize-2)
plt.ylabel('Absorbance', fontsize=fontsize-2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
# figure_4.savefig(Op_file+ str(r'\Spectra_visual')+".svg", bbox_inches='tight')



plt.show()