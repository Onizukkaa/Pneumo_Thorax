# -*- coding: utf-8 -*-

"""
Created on Thu Jun 23 09:18:35 2022

@author: joach
"""
import cv2
import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_addons as tfa
#%%
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#%%
BATCH_SIZE = 16
EPOCHS = 30
IM_SIZE_W = 300
IM_SIZE_H = 400 

AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.random.set_seed(10)


#%%
for dirname, _, filenames in os.walk('D:/programmation/Entrainement_perso/3_types_pneumo_ComputerVisio'):
    print(dirname)
    
#%%
"""ALL FILENAMES"""

filenames = tf.io.gfile.glob("D:/programmation/Entrainement_perso/3_types_pneumo_ComputerVisio/Data/*/*")
print(len(filenames))
print(filenames[:3])

#%%
"""TO DATAFRAME"""

data = pd.DataFrame()
for el in range(0, len(filenames)):
    target = filenames[el].split("\\")[-2]
    path = filenames[el]
    
    data.loc[el, 'filename'] = path
    data.loc[el, 'class'] = target

print(data['class'].value_counts(dropna=False)) 
    


#%%
"""SHUFFLE DATA"""

data = shuffle(data, random_state = 42)
data.reset_index(drop = True, inplace = True)
#%%
change = {
    
    "Normal" : "0",
    "Pneumonia-Bacterial" : "1",
    "Pneumonia-Viral" : "2",
    "COVID-19" : "3",
    }

data["class"] = data["class"].map(change)

#%%
"""IMAGES SHAPE"""

for i in range(100, 120):
    path = data.loc[i, "filename"]
    img = cv2.imread(path)
    print(img.shape)
    
#%%
"""SPLIT TRAIN_DATA, VAL_DATA"""

train_data, val_data = train_test_split(data, test_size = 0.2, random_state = 42, stratify = data["class"])
print(train_data["class"].value_counts(dropna = False))
print(val_data["class"].value_counts(dropna = False))

#%%
"""SPLIT TRAIN_DATA, TEST_DATA"""

train_data, test_data = train_test_split(train_data, test_size = 0.1, random_state = 42, stratify= train_data["class"])
print(train_data["class"].value_counts(dropna = False))
print(test_data["class"].value_counts(dropna = False))

#%%
"""DEFINe ImageDataGenerator and Augmentation(for train only!!)"""

datagen = ImageDataGenerator(rescale = 1.0/255,
                             zoom_range = 0.1,
                             brightness_range  = [0.9, 1.0],
                             height_shift_range = 0.05,
                             width_shift_range = 0.05,
                             rotation_range = 10,
                             )

test_datagen = ImageDataGenerator(rescale = 1.0/255)

train_gen = datagen.flow_from_dataframe(train_data, 
                                        x_col = "filename",
                                        y_col = "class",
                                        target_size = (IM_SIZE_W, IM_SIZE_H),
                                        color_mode = "grayscale",
                                        batch_size = BATCH_SIZE,
                                        class_mode = "categorical",
                                        shuffle = True,
                                        num_parallel_calls = AUTOTUNE
                                        )

val_gen = datagen.flow_from_dataframe(val_data, 
                                        x_col = "filename",
                                        y_col = "class",
                                        target_size = (IM_SIZE_W, IM_SIZE_H),
                                        color_mode = "grayscale",
                                        batch_size = BATCH_SIZE,
                                        class_mode = "categorical",
                                        shuffle = False,
                                        num_parallel_calls = AUTOTUNE
                                        )


test_gen = datagen.flow_from_dataframe(test_data, 
                                        x_col = "filename",
                                        y_col = "class",
                                        target_size = (IM_SIZE_W, IM_SIZE_H),
                                        color_mode = "grayscale",
                                        batch_size = BATCH_SIZE,
                                        class_mode = "categorical",
                                        shuffle = False,
                                        num_parallel_calls = AUTOTUNE
                                        )



#%%
"""CNN MODEL"""
def create_model():
    
    #model input
    input_layer = layers.Input(shape = (IM_SIZE_W, IM_SIZE_H, 1), name = "input")
    
    #forst block
    x = layers.Conv2D(filters= 128, kernel_size = 3, activation = "relu", padding = "same", name = "conv2d_1")(input_layer)
    x = layers.MaxPool2D(pool_size = 2, name = "maxpool2d_1")(x)
    x = layers.Dropout(0.1, name = "dropout_1")(x)
    
    # Second block
    x = layers.Conv2D(filters= 128, kernel_size = 3, activation = "relu", padding = "same", name = "conv2d_2")(x)
    x = layers.MaxPool2D(pool_size = 2, name = "maxpool2d_2")(x)
    x = layers.Dropout(0.1, name = "dropout_2")(x)
    
    # 3 block
    x = layers.Conv2D(filters= 128, kernel_size = 3, activation = "relu", padding = "same", name = "conv2d_3")(x)
    x = layers.MaxPool2D(pool_size = 2, name = "maxpool2d_3")(x)
    x = layers.Dropout(0.1, name = "dropout_3")(x)
    
    #4 block    
    x = layers.Conv2D(filters= 256, kernel_size = 3, activation = "relu", padding = "same", name = "conv2d_4")(x)
    x = layers.MaxPool2D(pool_size = 2, name = "maxpool2d_4")(x)
    x = layers.Dropout(0.1, name = "dropout_4")(x)
    
    # 5 block
    x = layers.Conv2D(filters= 256, kernel_size = 3, activation = "relu", padding = "same", name = "conv2d_5")(x)
    x = layers.MaxPool2D(pool_size = 2, name = "maxpool2d_5")(x)
    x = layers.Dropout(0.1, name = "dropout_5")(x)
    
    #6 block
    x = layers.Conv2D(filters= 512, kernel_size = 3, activation = "relu", padding = "same", name = "conv2d_6")(x)
    x = layers.MaxPool2D(pool_size = 2, name = "maxpool2d_6")(x)
    x = layers.Dropout(0.1, name = "dropout_6")(x)
    
    #7 block
    
    x = layers.Conv2D(filters= 512, kernel_size = 3, activation = "relu", padding = "same", name = "conv2d_7")(x)
    x = layers.MaxPool2D(pool_size = 2, name = "maxpool2d_7")(x)
    x = layers.Dropout(0.1, name = "dropout_7")(x)
    
    #• GlobalAveragePooling
    x = layers.GlobalAveragePooling2D(name = "global_average_pooling2d")(x)
    x = layers.Flatten()(x)
    
    #Head
    x = layers.Dense(1024, activation = "relu")(x)
    x = layers.Dropout(0.1, name = "dropout_head_2")(x)
    x = layers.Dense(128, activation = "relu")(x)
    
    #Output
    output = layers.Dense(units = 4,
                          activation = "softmax",
                          name = "output")(x)
    
    model = Model(input_layer, output)
    
    F_1_macro = tfa.metrics.f_scores.F1Score(num_classes = 4, average = "macro", name = "f1_macro", )
    
    model.compile(optimizer = "adam",
                  loss = "categorical_crossentropy",
                  metrics = F_1_macro)
    
    return model

model = create_model()

#%%
#model.summary()
init_time = datetime.datetime.now()

train_steps = train_gen.samples // BATCH_SIZE
valid_steps = val_gen.samples // BATCH_SIZE

early_stopping = EarlyStopping(monitor = "val_loss",
                               patience = 8,
                               mode = "min")

checkpoint = ModelCheckpoint("loss-{val_loss:.4f}.h5", 
                             monitor="val_loss", 
                             verbose=0, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             mode="min")

learning_rate_reduction = ReduceLROnPlateau(monitor = "val_loss",
                                            factor = 0.05,
                                            patience = 1,
                                            min_lr = 1e-7,
                                            verbose = 1,
                                            mode = "min")

history = model.fit(
    train_gen,
    validation_data = val_gen,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    steps_per_epoch = train_steps,
    callbacks = [
        early_stopping,
        checkpoint,
        learning_rate_reduction
        ],
    verbose = 1,
    )

requared_time = datetime.datetime.now() - init_time
print(f"\n Required Time : {str(requared_time)} \n")

#%%
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
#%%
test_steps = test_gen.samples // BATCH_SIZE

test_loss, test_acc = model.evaluate(test_gen, steps=test_steps)
print('\naccuracy:', test_acc, 'loss: ',test_loss)

#%%
predict = model.predict(test_gen, steps=test_steps)
y_hat = np.argmax(predict, axis=1)
print(y_hat[:20])

#%%

test_labels_df = pd.DataFrame()
test_labels_df[['class']] = test_data[['class']]

change = {
    '0' : 0,
    '1' : 1,
    '2' : 2,
    '3' : 3,
            }

test_labels_df['class'] = test_labels_df['class'].map(change)
test_labels_df = test_labels_df[ : test_steps*BATCH_SIZE]


y_test = np.array(test_labels_df['class'])
print(y_test[:20])

#%%
"""CLASSIFICATION REPORT"""

print(classification_report(y_test, y_hat), '\n')
cm = confusion_matrix(y_test, y_hat)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='.0f', cbar=False)

#%%
model.save('D:/programmation/Entrainement_perso/3_types_pneumo_ComputerVisio/Models/Main_2')
#%%
model = tf.keras.models.load_model("D:/programmation/Entrainement_perso/3_types_pneumo_ComputerVisio/Models/Main_2")
#%%
model.summary()


#%%
"""FEATURE VISUALIZATION """

import tf_keras_vis 
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
#%%
iterator = iter(test_gen)

imgs, labs = iterator.next()
real_labs = list(np.argmax(labs, axis = 1))
print(real_labs)
#%%
class_dict = {
    0 : 'Normal',
    1 : 'Pneumonia-Bacterial',
    2 : 'Pneumonia-Viral',
    3 : 'COVID-19',
                    }
#%%
BATCH_SIZE = 4

plt.rcParams['font.size'] = '20'
plt.subplots(BATCH_SIZE,2,figsize=(20,160))

idx=1

def get_gradcam_plus(img,
                    score,
                    model=model,
                    model_modifier=ReplaceToLinear()):
    
    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(model,
                          model_modifier=model_modifier,
                          clone=True)
    
    cam = gradcam(score,
                  img)
    
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    
    return heatmap



for i,img in enumerate(imgs[:BATCH_SIZE]):
    img_4d = tf.cast(tf.reshape(img, [1, IM_SIZE_W, IM_SIZE_H, 1]), tf.float32)
    predict = model.predict(img_4d)
    print(predict)
    prd = np.argmax(predict)
    print(f'class: {class_dict[prd]}')
    score1 = CategoricalScore(prd)
    original_lab = real_labs[i]
    
    plt.subplot(BATCH_SIZE,2,idx)
    plt.title(f'orignal {class_dict[original_lab]}')
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.binary)
    idx+=1
  
    plt.subplot(BATCH_SIZE,2,idx)
    gdcam_pls = get_gradcam_plus(img, score1)
    plt.imshow(img, cmap=plt.cm.binary)
    if prd:
        plt.imshow(gdcam_pls, alpha=0.2, cmap='jet')
    
    proba = round(float(predict[0][prd]), 4)
    plt.title(f'predicted {class_dict[prd]}  {proba} probability')
    plt.axis('off')
    idx+=1
    if idx>BATCH_SIZE*2:
        break

plt.tight_layout()
plt.show()




















