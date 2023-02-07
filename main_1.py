# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:18:35 2022

@author: joach
"""

import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tqdm import tqdm
#%%
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#%%
"""Read in images, equalize histogram and save to working_dir/histo"""

sdir = r"D:/programmation/Entrainement_perso/3_types_pneumo_ComputerVisio/Data"
working_dir = r"D:/programmation/Entrainement_perso/3_types_pneumo_ComputerVisio"
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
histpath = os.path.join(working_dir, "hist") #store equalized histogram images in this directory
#%%
if os.path.isdir(histpath):
    shutil.rmtree(histpath) #start with an empty directory

os.mkdir(histpath) #create directory
classlist = os.listdir(sdir) #iteratethrough the classes

for klass in classlist:
    print(f"processing class {klass}")
    classpath = os.path.join(sdir, klass)
    dest_classpath = os.path.join(histpath, klass)
    os.mkdir(dest_classpath) #Make class directories
    flist = os.listdir(classpath)
    sampled_list = np.random.choice(flist, size = 500, replace = False) #♣use only 500 images from each class
    
    for f in tqdm(sampled_list):
        fpath = os.path.join(classpath, f)
        dest_fpath = os.path.join(dest_classpath, f)
        img = cv2.imread(fpath, 0) #read in the image
        cl1 = clahe.apply(img) #Balance the image histogram
        cv2.imwrite(dest_fpath, cl1) #Save the balanced image
    
    
#%%
"""Read in images and create a dataframe of image paths and class labels"""

sdir = histpath
classlist = sorted(os.listdir(sdir))
filepaths = []
labels = []

for klass in classlist:
    classpath = os.path.join(sdir, klass)
    flist = sorted(os.listdir(classpath))
    
    for f in flist:
        fpath = os.path.join(classpath, f)
        filepaths.append(fpath)
        labels.append(klass)
        
Fseries = pd.Series(filepaths, name = "filepaths")
Lseries = pd.Series(labels, name = "labels")
df = pd.concat([Fseries, Lseries], axis = 1)
train_df, dummy_df = train_test_split(df, train_size = 0.9, shuffle = True, random_state= 123, stratify= df["labels"])
valid_df, test_df = train_test_split(dummy_df, train_size = 0.5, shuffle = True, random_state = 123, stratify = dummy_df["labels"])
print(f"train_df length : {len(train_df)}, test_df length : {len(test_df)}, valid_df length : {len(valid_df)}")

#get the number of classes and the images count for each class in train_df

classes = sorted(list(train_df["labels"].unique()))
class_counts = len(classes)
print(f"The number of classes in the dataset is : {class_counts}")
groups = train_df.groupby("labels")
print("{0:^30s} {1:^13s}".format("CLASS", "IMAGE COUNT"))

countlist = []
classlist = []

for label in sorted(list(train_df["labels"].unique())):
    group = groups.get_group(label)
    countlist.append(len(group))
    classlist.append(label)
    print("{0:^30s} {1:^13s}".format(label, str(len(group))))

#Get the class with the minimum and maximum number of train images

max_value = np.max(countlist)
max_index = countlist.index(max_value)
max_class = classlist[max_index]
min_value = np.min(countlist)
min_index = countlist.index(min_value)
min_class = classlist[min_index]
print(f"{max_class} has the most images = {max_value}, {min_class} has the least images = {min_value}")

# lets get the average height and width of sample of the train images

ht = 0
wt = 0

#Select 100 random samples of train_df

train_df_sample = train_df.sample(n = 100, random_state = 123, axis = 0)

for i in range(len(train_df_sample)):
    fpath = train_df_sample["filepaths"].iloc[i]
    img = plt.imread(fpath)
    shape= img.shape
    ht += shape[0]
    wt += shape[1]

print(f"average height = {ht//100}, average width = {wt//100}, aspect ratio = {ht/wt}")    
    
#%%
"""Create the train_gen, test_gen final_test_gen and valid_gen"""

img_size = (300, int(300/0.8))
batch_size = 4 # We will use and EfficientB3 modeln with image size of (200,250) this size should not cause resource error
trgen = ImageDataGenerator(horizontal_flip = True,
                           rotation_range = 20,
                           width_shift_range = 0.2,
                           height_shift_range = 0.2, 
                           zoom_range = 0.2)

t_and_v_gen = ImageDataGenerator()
msg = '{0:70s} for train generator'.format(' ')
print(msg, "\r", end="") #Print over on the same line

train_gen = trgen.flow_from_dataframe(train_df, x_col = "filepaths", y_col = "labels", target_size = img_size,
                                      class_mode = "categorical", color_mode = "rgb", shuffle = True, batch_size = batch_size)

msg='{0:70s} for valid generator'.format(' ')
print(msg, "\r", end="")

valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col = "filepaths", y_col = "labels", target_size = img_size,
                                      class_mode = "categorical", color_mode = "rgb", shuffle = False, batch_size = batch_size)

#for the tes_gen we want to calculate the batch size and test steps such that batch_size X test_steps = number of samples in test set
#This insures that we go through all the sample in the test set exactly once

length = len(test_df)
test_batch_size = sorted([int(length/n) for n in range(1, length + 1) if length % n == 0 and length/n <= 80], reverse = True)[0]
test_steps = int(length/test_batch_size)

msg='{0:70s} for test generator'.format(' ')
print(msg, '\r', end='') # prints over on the same line

test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col = "filepaths", y_col = "labels", target_size = img_size,
                                      class_mode = "categorical", color_mode = "rgb", shuffle = False, batch_size = batch_size)


# From the generator we can get information we will need later
classes = list(train_gen.class_indices.keys())
class_indices = list(train_gen.class_indices.values())
class_count = len(classes)
labels = test_gen.labels

print(f"Test batch size : {test_batch_size}, test steps : {test_steps}, number of classes : {class_count}")

#%%
"""Create a function to show example training images"""

def show_images_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen) # get a sample batch from the generator
    plt.figure(figsize = (20,20))
    length = len(labels)
    
    if length < 25 : #show maximum of 25 images
        r = length
    else:
        r = 25
        
    for i in range(r):
        plt.subplot(5, 5, i+1)
        image = images[i] / 255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color = "blue", fontsize = 12)
        plt.axis("off")
    plt.show()
    
show_images_samples(train_gen)

#%%
"""Create a model using transfer learning with EfficientNetB3"""

img_shape = (img_size[0], img_size[1], 3)
model_name = "EfficientNetB3"

base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top = False,
                                                               weights = "imagenet",
                                                               input_shape = img_shape,
                                                               pooling = "max")

#Note your are always told NOT to make the base model trainable initially, that is WRONG you gey better results leaing it trainable

base_model.trainable = True
x = base_model.output
x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016), activity_regularizer = regularizers.l1(0.006),
          bias_regularizer = regularizers.l1(0.006), activation = "relu")(x)
x = Dropout(rate = 0.4, seed = 123)(x)
output = Dense(class_count, activation = "softmax")(x)

model = Model(inputs = base_model.input, outputs = output)
lr = 0.001 #@start with this lr

model.compile(Adamax(learning_rate = lr), 
              loss = "categorical_crossentropy", 
              metrics = ["accuracy"])



#%%
"""Instantiate custom callback and create 2 callback to control learning rate and early stop"""

epochs = 40

rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.05, patience = 2, verbose = 1)
estop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 6, verbose = 1, restore_best_weights = True)
callbacks = [rlronp, estop]

"""Train the model"""

history = model.fit(x = train_gen,
                    epochs = epochs,
                    verbose = 1,
                    callbacks = callbacks,
                    validation_data = valid_gen,
                    validation_steps = None,
                    shuffle = False,
                    initial_epoch = 0) 

#%%
model.summary()
#%%
"""Define a fonction to plot the training data"""

def tr_plot(tr_data, start_epoch):
    #Plot the training and validation data
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(tacc)+ start_epoch
    
    Epochs = []
    
    for i in range (start_epoch ,Epoch_count):
        Epochs.append(i+1)   
        
    index_loss = np.argmin(vloss)#  this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig,axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20,8))
    
    axes[0].plot(Epochs,tloss, 'r', label = 'Training loss')
    axes[0].plot(Epochs,vloss,'g',label = 'Validation loss' )
    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s = 150, c = 'blue', label = sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label = 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label = 'Validation Accuracy')
    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s = 150, c = 'blue', label = vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout    
    plt.show()
    
tr_plot(history,0)




#%%
""""Make Predictions on the test set </a>"""


def predictor(test_gen, test_steps):
    y_pred = []
    y_true = test_gen.labels
    classes = list(train_gen.class_indices.keys())
    class_count = len(classes)
    errors = 0
    preds=model.predict(test_gen, steps=test_steps, verbose=1) # predict on the test set
    tests = len(preds)
    for i, p in enumerate(preds):
            pred_index=np.argmax(p)         
            true_index=test_gen.labels[i]  # labels are integer values
            
            if pred_index != true_index: # a misclassification has occurred                                           
                errors=errors + 1
            y_pred.append(pred_index)
            
    acc=( 1-errors/tests) * 100
    print(f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}')
    ypred=np.array(y_pred)
    ytrue=np.array(y_true)
    
           
       
    #clr = classification_report(y_true, y_pred, target_names=classes, digits= 4) # create classification report
    #print("Classification Report:\n----------------------\n", clr)
    
    
    return errors, tests


errors, tests=predictor(test_gen, test_steps)

#%%
"""Save the model"""

subject = 'pneumonia' 
acc = str(( 1-errors/tests) * 100)
index = acc.rfind('.')
acc = acc[:index + 3]
save_id = subject + '_' + str(acc) + '.h5' 
model_save_loc = os.path.join(working_dir, save_id)
model.save(model_save_loc)
print ('model was saved as ' , model_save_loc ) 
   

#%%
predict = model.predict(test_gen, steps = test_steps)
y_hat = np.argmax(predict, axis = 1)
print(y_hat[:20])








    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    