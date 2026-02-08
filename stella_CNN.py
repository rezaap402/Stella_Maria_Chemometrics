import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, Flatten, LSTM
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
import tensorflow.keras.metrics
from sklearn.metrics import mean_squared_error, r2_score
# from livelossplot import PlotLossesKerasTF
from sklearn.svm import SVR
from Preprocessings import snv, msc, sg1



SMALL_SIZE = 12
MEDIUM_SIZE = SMALL_SIZE+6
BIGGER_SIZE = SMALL_SIZE+8

plt.rc('text', usetex=True)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title\




def reproducible_comp():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(42)

reproducible_comp()
# import the files
wavelength = np.arange(900, 2200+1) # check again, idk this is true or not kwkw

Op_file = 'E:\FROM COMPUTERS_26 DES 2025\MyPhD\Stella_data';

# calibration
Cfile = pd.read_excel(Op_file+ str(r'\M10-v2_Cal70')+".xlsx")
X_train = Cfile.values[:, 1:]; # if your data is not % reflectance, just delete the *100
y_train = Cfile.values[:, 0];

# prediction/validation
Pfile = pd.read_excel(Op_file+ str(r'\M10-v2_Pred30')+".xlsx");
X_test = Pfile.values[:, 1:];
y_test = Pfile.values[:, 0];

X_train = sg1(X_train)
X_test = sg1(X_test)

# for training and validating
X_Train1, X_Test1, Y_Train_, Y_Test_ = train_test_split(X_train, y_train, test_size=0.20, random_state=42) # or 0.20


reproducible_comp() # for stability

# this is 1D
length = X_Train1.shape[1]
input_shape = (length,1)

def CornNet():
  K_NUMBER = 1
  K_WIDTH = 5 # kernel_size
  K_STRIDE = 1

  # input layer
  inputs = Input(shape=input_shape,name='input')
  # convolutional layer 1st
  x = Conv1D(32, kernel_size=K_WIDTH, strides=K_STRIDE, padding='same', kernel_initializer='he_normal', name='CONV_1D-1')(inputs)
  x = MaxPooling1D(pool_size=2, strides=K_STRIDE, name='MaxPOOL-1')(x)
  x = Activation('elu')(x)
  # layer
  # x = Conv1D(64, kernel_size=K_WIDTH, strides=K_STRIDE, padding='same', kernel_initializer='he_normal', name='CONV_1D-1')(inputs)
  # x = MaxPooling1D(pool_size=2, strides=K_STRIDE, name='MaxPOOL-1')(x)
  # x = Activation('elu')(x)
  # flatten layer
  x = Flatten()(x)
  # fully connected layers (this is ANN)
  x = Dense(64, name='FC_1')(x)
  x = Activation('elu')(x)
  x = Dense(32, name='FC_2')(x)
  x = Activation('elu')(x)
  x = Dense(4, name='FC_3')(x)
  x = Activation('elu')(x)
  # regression layer
  outputs = Dense(1, activation='linear', name='Regression_layer')(x)
  return Model(inputs=inputs, outputs=outputs)

"Configurations for the 1D Network in Regression Mode"
model = CornNet()

model.summary() # print out

#######  Callbacks #########

tf.keras.backend.clear_session()

hname = 'D:\MyPhD\Stella_data\Saved_Corn-v2'
model_loc = Op_file + str(r'\Deep_Model_sg1.h5')

early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, mode='auto', restore_best_weights=True)
rdlr = ReduceLROnPlateau(patience=25, factor=0.5, min_lr=1e-6, monitor='val_loss', verbose=0)
checkpointer = ModelCheckpoint(filepath=model_loc, verbose=1, save_best_only=True)
# plot_losses = PlotLossesKerasTF()

BATCH = 8
LR = 0.01*BATCH/256.
EPOCHS = 500

optimizer = tf.keras.optimizers.Adam(learning_rate = LR)
model.compile(optimizer=optimizer, loss="mse", metrics=['mse'])

history0 = model.fit(X_Train1, Y_Train_, batch_size=BATCH, epochs=EPOCHS, \
          validation_data=(X_Test1, Y_Test_),\
          callbacks=[checkpointer, rdlr, early_stop], verbose=1)

hist_df = pd.DataFrame(history0.history)
hist_df.to_excel(Op_file + str(r'\Deep_History_sg1.xlsx'), index=False)

tf.keras.backend.clear_session()

# for history visualization
f2=plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(history0.history['loss'], label='Training')
plt.plot(history0.history['val_loss'], label='Tunning')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(linestyle='--', alpha=0.2)
plt.subplot(1,2,2)
plt.plot(history0.history['mse'], color='b', label='Training')
plt.plot(history0.history['val_mse'], color='orange', label='Tunning')
plt.ylabel('MSE (\%)')
plt.xlabel('Epochs')
plt.legend()
plt.grid(linestyle='--', alpha=0.2)
plt.tight_layout()
# for saving the images, activate when u wanna download the imgs
# f2.savefig('LOSSPLOT_Coklat.png', format='png', dpi=500)

# make the regression line
def regline(true,pred):
  rangey = max(true) - min(pred)
  rangex = max(pred) - min(pred)
  z = np.polyfit(np.ravel(true), np.ravel(pred), 1)
  z1 = z[1]+z[0]*true
  return z1

# turns back for calculation
model_base = load_model(model_loc, compile=False) # calling from the pretrained
C_Y_values = model_base.predict(X_train)

R2c =  r2_score(y_train, C_Y_values)
# if rmse, choose one
RMSEC = (mean_squared_error(y_train, C_Y_values))**0.5;
# if sec
# eC = (y_train-C_Y_values)
# SEC = np.std(eC)

V_Y_values = model_base.predict(X_test)
R2p = r2_score(y_test, V_Y_values)
RMSEP = (mean_squared_error(y_test, V_Y_values))**0.5;
# if wanna use SEP just follow the previous code
RPD = 1/((1-R2p)**0.5);
LOQ = 10*RMSEP;
LOD = 3.3*RMSEP;

# visualize
f3 = plt.figure(figsize=(6,4))
plt.scatter(y_train, C_Y_values,c='blue',s=10, label='Calibration')
plt.plot(y_train, regline(y_train, C_Y_values), c='b', linewidth=1.5, label='Reg line (Calibration)')
plt.scatter(y_test, V_Y_values,c='orange',s=10, label='Prediction')
plt.plot(y_test, regline(y_test, V_Y_values), c='orange', linewidth=1.5, label='Reg line (Prediction)')
plt.xlabel('Actual moisture content (\%wb)')
plt.ylabel('Predicted moisture content (\%wb)')
plt.tight_layout()
plt.legend()

print('===== CALIBRATION =====')
print(f'R2c: {R2c:.3f}')
print(f'RMSEC: {RMSEC:.3f} %wb')
print('==== PREDICTION ====')
print(f'R2p: {R2p:.3f}')
print(f'RMSEP: {RMSEP:.3f} %wb')
print(f'RPD: {RPD:.3f}')
print(f'LOQ: {LOQ:.3f} %wb')
print(f'LOD: {LOD:.3f} %wb')

plt.show()

# for saving the images, activate when u wanna download the imgs
# f3.savefig('SCATTER_Coklat.png', format='png', dpi=500)

# extract the features from the deep leaerning
# feature_extractor = Model(
#     inputs=model_base.input,
#     outputs=model_base.get_layer('flatten').output
# )
# feat_train = feature_extractor.predict(X_Train1)

# svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1) # just use the fixed one, no need to optimize

# n_train = feat_train.shape[0]
# feat_train_flat = feat_train.reshape(n_train, -1)
# svr_rbf.fit(feat_train, Y_Train_)

# C_Y_values2 = svr_rbf.predict(feature_extractor.predict(X_train))
# R2c =  r2_score(y_train, C_Y_values2)
# # if rmse, choose one
# RMSEC = (mean_squared_error(y_train, C_Y_values2))**0.5;
# # if sec
# # eC = (y_train-C_Y_values2)
# # SEC = np.std(eC)

# V_Y_values2 = svr_rbf.predict(feature_extractor.predict(X_test))
# R2p = r2_score(y_test, V_Y_values)
# RMSEP = (mean_squared_error(y_test, V_Y_values2))**0.5;
# # if wanna use SEP just follow the previous code
# RPD = 1/((1-R2p)**0.5);
# LOQ = 10*RMSEP;
# LOD = 3.3*RMSEP;

# f4 = plt.figure(figsize=(6,4))
# plt.scatter(y_train, C_Y_values2,c='blue',s=10, label='Calibration')
# plt.plot(y_train, regline(y_train, C_Y_values2), c='b', linewidth=1.5, label='Reg line (Calibration)')
# plt.scatter(y_test, V_Y_values2,c='orange',s=10, label='Prediction')
# plt.plot(y_test, regline(y_test, V_Y_values2), c='orange', linewidth=1.5, label='Reg line (Prediction)')
# plt.xlabel('Actual moisture content (%wb)')
# plt.ylabel('Predicted moisture content (%wb)')
# plt.legend()

# print('===== CALIBRATION =====')
# print(f'R2c: {R2c:.3f}')
# print(f'RMSEC: {RMSEC:.3f} %wb')
# print('==== PREDICTION ====')
# print(f'R2p: {R2p:.3f}')
# print(f'RMSEP: {RMSEP:.3f} %wb')
# print(f'RPD: {RPD:.3f}')
# print(f'LOQ: {LOQ:.3f} %wb')
# print(f'LOD: {LOD:.3f} %wb')

# for saving the images, activate when u wanna download the imgs
# f4.savefig('SCATTER_Coklat.png', format='png', dpi=500)

# for saving the model