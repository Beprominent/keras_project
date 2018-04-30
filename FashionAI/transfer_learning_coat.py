import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('/home/neuiva1/xie/data/FashionTag/base/Annotations/label.csv', header=None)
df_train.columns = ['image_id', 'class', 'label']
print(df_train.head())

classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels',
           'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels',
           'pant_length_labels']

cur_class = classes[5]
df_load = df_train[(df_train['class']) == cur_class].copy()
df_load.reset_index(inplace=True)
del df_load['index']
print(df_load)

print('{0}: {1}'.format(cur_class, len(df_load)))
print(df_load.head())
width = 299

n = len(df_load)
n_classes = len(df_load['label'][0])
print(n_classes)

X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_classes), dtype=np.uint8)
print('------')

for i in tqdm(range(n)):
    tmp_label = df_load['label'][i]
    if len(tmp_label) > n_classes:
        print(df_load['image_id'][i])
    print(df_load['image_id'][i])
    X[i] = cv2.resize(cv2.imread('/home/neuiva1/xie/data/FashionTag/base/{0}'.format(df_load['image_id'][i])), (width, width))
    y[i][tmp_label.find('y')] = 1

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.applications.inception_resnet_v2 import preprocess_input


cnn_model = InceptionResNetV2(include_top=False, input_shape=(width, width, 3), weights='imagenet')
inputs = Input((width, width, 3))
print("-----")
x = inputs
x = Lambda(preprocess_input, name='preprocessing')(x)
x = cnn_model(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(n_classes, activation='sigmoid', name='sigmoid')(x)
print("-----")
model = Model(inputs, x)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.12, random_state=42)


prefix_cls = cur_class.split('_')[0]

model.compile(optimizer='adadelta',
             loss='binary_crossentropy',
             metrics=['accuracy'])

filepath = '{0}.best.h5'.format(os.path.join("/home/neuiva1/xie/Kaggle/FashionTag2/sigmoid", prefix_cls))
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                               save_best_only=True)
h = model.fit(X_train, y_train, batch_size=16, epochs=80,
             shuffle=True,
             validation_split=0.1)
with open('/home/neuiva1/xie/Kaggle/FashionTag2/sigmoid/log_coat.txt', 'w') as f:
	f.write(str(h.history))


model.evaluate(X_train, y_train, batch_size=256)
model.evaluate(X_valid, y_valid, batch_size=256)


df_test = pd.read_csv('/home/neuiva1/xie/data/FashionTag/rank/Tests/question.csv', header=None)
df_test.columns = ['image_id', 'class', 'x']
del df_test['x']

df_load = df_test[(df_test['class'] == cur_class)].copy()
df_load.reset_index(inplace=True)
del df_load['index']

print('{0}: {1}'.format(cur_class, len(df_load)))
df_load.head()

n = len(df_load)
X_test = np.zeros((n, width, width, 3), dtype=np.uint8)

for i in tqdm(range(n)):
    X_test[i] = cv2.resize(cv2.imread('/home/neuiva1/xie/data/FashionTag/rank/{0}'.format(df_load['image_id'][i])), (width, width))

test_np = model.predict(X_test, batch_size=256)
print(test_np.shape)

result = []

for i, row in df_load.iterrows():
    tmp_list = test_np[i]
    tmp_result = ''
    for tmp_ret in tmp_list:
        tmp_result += '{:.4f};'.format(tmp_ret)

    result.append(tmp_result[:-1])

df_load['result'] = result
df_load.head()

result_file = "/home/neuiva1/xie/Kaggle/FashionTag2/sigmoid/result/{}.csv".format(prefix_cls)

df_load.to_csv(result_file, header=None, index=False)



