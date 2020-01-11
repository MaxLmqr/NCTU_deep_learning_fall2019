import pickle
import sys
import numpy as np
from keras.utils import to_categorical



"""
        ['airplane', 
        'automobile', 
        'bird', 
        'cat', 
        'deer', 
        'dog', 
        'frog', 
        'horse', 
        'ship', 
        'truck']
"""
#### PREPROCESSING THE DATA ##############################
def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
        data = d['data']
        labels = d['labels']
        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels



for i in range(1,6):
    file = 'data_batch_'+str(i)
    temp_feature, temp_label = unpickle(file)    # Tuple containing data and label : len=2
    index_val = int(len(temp_feature)*0.1)
    if i==1:
        feature = temp_feature[:10000-index_val]
        val_feature = temp_feature[-index_val:]
        label = temp_label[:10000-index_val]
        val_label = temp_label[-index_val:]
    else:
        feature = np.concatenate((feature, temp_feature[:10000-index_val]))
        val_feature = np.concatenate((val_feature,temp_feature[-index_val:]),axis=0)
        label = np.concatenate((label, temp_label[:10000-index_val]),axis=0)
        val_label = np.concatenate((val_label,temp_label[-index_val:]),axis=0)

feature = feature.transpose(0,2,3,1)
val_feature = val_feature.transpose(0,2,3,1)

feature = feature/255
val_feature = val_feature/255
label = to_categorical(label,num_classes=10)
val_label = to_categorical(val_label, num_classes=10)

pickle.dump((feature,label),open('preprocess.p','wb'))
pickle.dump((val_feature,val_label),open('preprocess_validation.p','wb'))