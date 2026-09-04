# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
# This script takes aggregate statistics and
# gets asd predictions
# April 2022 : Converted from colab notebook to python file by Sidrah

# Import libraries
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import sys
import scipy as sp
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import AlphaDropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


def plot_acc_loss(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


version = str(sys.argv[1])
DATAPATH = 'PATH/TO/FEATURES/aggregate/'
ados_sarb = pd.read_csv('aggregated_social_scores_08182021.csv')
ados_sarb.filename = ados_sarb.filename.str.replace('.mpg', '.csv')
#DATAPATH = 'PATH/TO/FEATURES/aggregate_fromoct2023/aggregate/'
# Load the combined Parent and Examiner dataset and drop unwanted columns
test_data_path = DATAPATH + 'hj_test_{}_Aggregated.csv'.format(version)
train_data_path = DATAPATH + 'hj_asdtrain_{}_Aggregated.csv'.format(version)
all_data_list = (pd.read_csv(file) for file in [test_data_path, train_data_path])
all_data = pd.concat(all_data_list, ignore_index=True)

# Initialize empty lists to store the data
spec_list = []
sens_list = []
prec_list = []
f1_list = []
acc_list = []
aucroc_list = []

test_result = pd.DataFrame(columns = ['filename', 'sub_id', 'visit', 'asd_longitudinal', 'pred_score', 'pred_prob'])

all_data = pd.merge(all_data, ados_sarb[['filename','ados_sarb_total']], on=['filename'], how='left')

fold = 3
for subfold in range(10):
    print('Subfold {}'.format(subfold))
    if subfold == 3:
        continue
    fname_test = 'PATH/TO/FEATURES/data_splits/standard_filenames/fold{}/{}_fold{}_subfold{}.txt'.format(fold, 'test', fold,
                                                                                               subfold)
    fname_train = 'PATH/TO/FEATURES/data_splits/standard_filenames/fold{}/{}_fold{}_subfold{}.txt'.format(fold, 'asdtrain', fold,
                                                                                                subfold)
    # fname_behavior = ''
    # write python script to read a csv containing filenames, and get all rows in all_data_path where filename column matches those from csv file
    # filenames = pd.read_csv(fname, header=None)
    # filenames[0] = filenames[0]+'.csv'
    # filenames.dropna(inplace=True)

    # import the datasets
    fname_test_read = pd.read_csv(fname_test, header=None)
    fname_test_read[0] = fname_test_read[0] + '.csv'
    fname_test_read.dropna(inplace=True)
    data_test = all_data[all_data.filename.astype(str).isin(fname_test_read[0])]
    # data_test = pd.read_csv(test_data_path)
    fname_train_read = pd.read_csv(fname_train, header=None)
    fname_train_read[0] = fname_train_read[0] + '.csv'
    fname_train_read.dropna(inplace=True)
    data_train = all_data[all_data.filename.astype(str).isin(fname_train_read[0])]
    # data_train = pd.read_csv(train_data_path)
    # convert to pd dataframe
    data_test_selected = pd.DataFrame(data_test)
    data_train_selected = pd.DataFrame(data_train)

    train = data_train_selected
    test = data_test_selected

    """---"""
    # train.drop(train[train.visit < 12].index, inplace=True)
    # test.drop(test[test.visit < 12].index, inplace=True)
    # map train for gender
    train.gender = train.gender.map({'Female': 0, 'Male': 1})
    # map test for gender
    test.gender = test.gender.map({'Female': 0, 'Male': 1})
    # map train for asd_longitudinal
    train.asd_longitudinal = train.asd_longitudinal.map({'Non-ASD': 0, 'ASD': 1})
    # map test for asd_longitudinal
    test.asd_longitudinal = test.asd_longitudinal.map({'Non-ASD': 0, 'ASD': 1})

    input_features_minus_weight = ['visit', 'gender',
                                   'total_smile_fq', 'smile_rate', 'total_smile_duration', 'smile_prop',
                                   'total_lookface_fq', 'lookface_rate', 'total_lookface_dur',
                                   'lookface_prop', 'total_lookobject_fq', 'lookobject_rate',
                                   'total_lookobject_dur', 'lookobject_prop', 'total_vocal_fq',
                                   'vocal_rate', 'total_vocal_duration', 'vocal_prop',
                                   'total_social_smile_fq', 'social_smile_rate',
                                   'total_social_smile_duration', 'social_smile_prop',
                                   'total_social_vocal_fq', 'social_vocal_rate',
                                   'total_social_vocal_duration', 'social_vocal_prop'
                                   ]
    """
    input_features_minus_weight = ['visit', 'gender',
                                    'smile_rate',  'smile_prop',
                                    'lookface_rate', 'lookface_prop',
                                    'lookobject_rate',
                                    'lookobject_prop',

                                   'vocal_rate',  'vocal_prop',
                                    'social_smile_rate',
                                    'social_smile_prop',
                                    'social_vocal_rate',
                                    'social_vocal_prop'
                                   ]
    """
    """
    input_features_minus_weight = [
        'smile_rate', 'smile_prop',
        'lookface_rate', 'lookface_prop',
        'social_smile_rate', 'social_smile_prop',
        'social_vocal_rate', 'social_vocal_prop'
    ]
    """
    X_train = train[input_features_minus_weight].values
    X_test = test[input_features_minus_weight].values

    y_train = train.loc[:, 'asd_longitudinal'].values
    y_test = test.loc[:, 'asd_longitudinal'].values

    # Apply SMOTE(Oversampling) followed by Tomeklinks(Undersampling)
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_resample(X_train, y_train)
    smt = SMOTETomek()
    X_smt, y_smt = smt.fit_resample(X_ros, y_ros)

    # Convert class vector (asd_new) to binary class matrix
    y_train2 = to_categorical(y_train)
    y_test2 = to_categorical(y_test)
    y_smt2 = to_categorical(y_smt)

    # Normalize All data (except for the output labels(y values))
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    X_smt = sc.fit_transform(X_smt)

    positive_count = np.count_nonzero(y_train == 1)
    negative_count = np.count_nonzero(y_train == 0)

    weight_for_0 = (1 / negative_count) * (y_train.shape[0] / 2)
    weight_for_1 = (1 / positive_count) * (y_train.shape[0] / 2)

    class_weights = {0: weight_for_0, 1: weight_for_1}

    # add loop to train the model n_times, and average the metrics over n_times
    for i in range(1):
        # define the keras model
        model = Sequential()
        model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
        model.add(AlphaDropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(AlphaDropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(AlphaDropout(0.5))
        model.add(Dense(2, activation='softmax'))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=90,
            decay_rate=0.9)

        opt = keras.optimizers.Adam(learning_rate=0.0009)

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='val_loss', patience=200, verbose=0, mode='min')
        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=1, epsilon=1e-4,
                                           mode='min')

        # mlp nn after overunder sampling
        #history = model.fit(X_smt, y_smt2, epochs=1000, batch_size=64, validation_data=(X_test, y_test2),
        #                    callbacks=[earlyStopping, mcp_save, reduce_lr_loss], verbose=0)
        # history = model.fit(X_train, y_train2, epochs=1000
        #                    , batch_size=64, validation_data = (X_test,y_test2),class_weight=class_weights,verbose=0,
        #                    callbacks = [earlyStopping, mcp_save, reduce_lr_loss])

        #model.save('hj_prev/hj_{}_ASD_model_subfold{}_{}.h5'.format(version, subfold, i))
        scores = model.predict(X_test)
        predicted_classes = np.argmax(scores, axis=-1)
        temp = pd.DataFrame(columns=['filename', 'sub_id', 'visit', 'asd_longitudinal', 'pred_score', 'pred_prob'])
        test.reset_index(inplace=True)
        temp.loc[:, ['filename', 'sub_id', 'visit', 'asd_longitudinal']] = test.loc[:, ['filename', 'sub_id', 'visit',
                                                                                        'asd_longitudinal']]
        # test_result=
        temp.pred_prob = pd.Series(scores[:,1])
        # scores.shape
        temp.pred_score = pd.Series(predicted_classes)
        test_result = pd.concat([test_result, temp])

        # ROC AUC
        auc = roc_auc_score(y_test, predicted_classes)
        print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(y_test, predicted_classes)
        print('Confusion Matrix:\n', matrix)
        report = classification_report(y_test, predicted_classes, output_dict=True)

        # import keras.models

        # model1 = keras.models.load_model('hj_F_ASD_model_9.h5')
        # predicted_classes = np.argmax(model1.predict(X_test), axis=-1)
        ind = np.where(((y_test != predicted_classes) & (y_test == 1)))
        abc = test.iloc[ind]
        print(abc.filename)

        spec_list.append(round(report['0']['recall'], 2))
        sens_list.append(round(report['1']['recall'], 2))
        prec_list.append(round(report['1']['precision'], 2))
        f1_list.append(round(report['macro avg']['f1-score'], 2))
        acc_list.append(round(report['accuracy'], 2))
        aucroc_list.append(round(auc, 2))

        # print(spec_list)
        # print(sens_list)
        # print(prec_list)
        # print(f1_list)
        # print(acc_list)
        # print(aucroc_list)

        # fpr, tpr, thresholds = roc_curve(y_test, predicted_classes,
        #                                 pos_label=1)  # here the second argument is not the LABELS, rather the softmax output values from the classifier
        # plot_roc_curve(fpr, tpr)
        # plot_acc_loss(history)
test_result.to_csv('save_output.csv', index=False)
visits = [12, 18, 24, 36]
for v in visits:

    print('At visit {}'.format(v))
    idx = test_result[test_result.visit == v].index

    #print(test_result.filename[idx])
    label = test_result.asd_longitudinal[idx].values
    prob = test_result.pred_prob[idx].values
    pred = test_result.pred_score[idx].values
    # ROC AUC
    auc = roc_auc_score(label.astype(int),pred.astype(int))
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(label.astype(int), pred.astype(int))
    print('Confusion Matrix:\n', matrix)
    report = classification_report(label.astype(int), pred.astype(int), output_dict=True)

    print('spec:{}'.format(round(report['0']['recall'], 2)))
    print('sens:{}'.format(round(report['1']['recall'], 2)))
    print('prec:{}'.format(round(report['1']['precision'], 2)))
    print('f1 score:{}'.format(round(report['macro avg']['f1-score'], 2)))
    print('acc:{}'.format(round(report['accuracy'], 2)))

# forming new dataframe
#test_result = pd.merge(test_result, ados_sarb, on=['filename'], how='left')
import re
test_result.reset_index(inplace=True)
test_result['partner'] = None
for r in range(test_result.shape[0]):
    fname = re.split('_|\.', str(test_result.filename[r]))
    test_result.loc[r, 'partner'] = fname[2]

visit_range = set(test_result.visit)

partner_range = set(test_result.partner)
col_names = [f'{visit}_{syn}' for visit in visit_range for syn in partner_range]
test_pivoted = test_result.pivot_table( index=['sub_id'], columns=[ 'visit', 'partner'], values=['pred_score', 'asd_longitudinal'], aggfunc='first')

#test_pivoted.columns = col_names

test_pivoted.reset_index(inplace=True)

test_pivoted.to_csv('age_wise.csv', index=None)


# Save the data to a file using pickle
with open("hj_prev/metrics_data_{}.pkl".format(version), "wb") as file:
    data_to_save = {
        "spec": spec_list,
        "sens": sens_list,
        "prec": prec_list,
        "f1": f1_list,
        "acc": acc_list,
        "aucroc": aucroc_list,
    }
    pickle.dump(data_to_save, file)

# Reading the data from the file
with open("hj_prev/metrics_data_{}.pkl".format(version), "rb") as file:
    loaded_data = pickle.load(file)
# Access the loaded data
loaded_spec = loaded_data["spec"]
loaded_sens = loaded_data["sens"]
loaded_prec = loaded_data["prec"]
loaded_f1 = loaded_data["f1"]
loaded_acc = loaded_data["acc"]
loaded_aucroc = loaded_data["aucroc"]

# Now you can work with the loaded data as needed
print(loaded_spec)
print(loaded_sens)
print(loaded_prec)
print(loaded_f1)
print(loaded_acc)
print(loaded_aucroc)

import numpy as np

print('sens')
print(np.mean(loaded_sens))
print(np.std(loaded_sens, ddof=1))
print('spec')
print(np.mean(loaded_spec))
print(np.std(loaded_spec, ddof=1))
print('prec')
print(np.mean(loaded_prec))
print(np.std(loaded_prec, ddof=1))
print('f1')
print(np.mean(loaded_f1))
print(np.std(loaded_f1, ddof=1))
print('aucroc')
print(np.mean(loaded_aucroc))
print(np.std(loaded_aucroc, ddof=1))
print('acc')
print(np.mean(loaded_acc))
print(np.std(loaded_acc, ddof=1))

print('pause')
"""# References
1. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
"""
"""
spec_ave = []
sens_ave = []
prec_ave = []
f1_ave = []
acc_ave = []
aucroc_ave = []
for i in range(9):
    spec_ave.append(np.mean(loaded_spec[i * 10:(i + 1) * 10]))
    sens_ave.append(np.mean(loaded_sens[i * 10:(i + 1) * 10]))
    f1_ave.append(np.mean(loaded_f1[i * 10:(i + 1) * 10]))
    aucroc_ave.append(np.mean(loaded_aucroc[i * 10:(i + 1) * 10]))
    acc_ave.append(np.mean(loaded_acc[i * 10:(i + 1) * 10]))
    prec_ave.append(np.mean(loaded_prec[i * 10:(i + 1) * 10])+0.4)
#plt.plot(loaded_prec[i * 10:(i + 1) * 10])
#plt.plot(loaded_sens[i * 10:(i + 1) * 10])
plt.plot(spec_ave)
plt.plot(sens_ave)
plt.plot(f1_ave)
plt.plot(aucroc_ave)
plt.plot(acc_ave)
plt.plot(prec_ave)
#plt.plot(prec_ave)
#plt.plot(loaded_f1[i * 10:(i + 1) * 10])
#plt.plot(loaded_aucroc[i * 10:(i + 1) * 10])
#plt.plot(loaded_acc[i * 10:(i + 1) * 10])
plt.title('Model Performance')
plt.legend(['Spec', 'Sens', 'F1 Score', 'AUCROC', 'Accuracy', 'Prec' ], loc='upper left')
plt.show()
"""

"""
import pickle as pkl
import pandas as pd
with open("file.pkl", "rb") as f:
    object = pkl.load(f)

df = pd.DataFrame(object)
df.to_csv(r'file.csv')

"""""