from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import Flatten,Dense,Dropout
from keras.preprocessing.image import img_to_array
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
#######################################################################################################################
#######################################################################################################################
#                                           ПАРАМЕТРЫ ДЛЯ ИЗМЕНЕНИЯ!!!!!!!!!!!!!!!
path=r"C:\Users\Dmitry\Desktop\CNN_Research_2018\cats_vs_dogs"
image_size=(150, 150) # здесь задавать размер входных изображений
batch_size=32 # выбирать из этих значений [8,16,32]
batch_size2=100
inception_nontrainable_layers_count=205 # количество слоёв InceptionV3, чьи веса мы не меняем при обучении, в процессе переноса обучения(transfer learning)
nb_epoch=1# количество эпох обучения нейронной сети
fc_nb_epoch=10# количество эпох обучения классификационной части сети
n_classes=2# количество классов для обучения
train_path=os.path.join(path,"test")# FIXME train!!!
validation_path=os.path.join(path,'validation')
test_path=os.path.join(path,'test')
path_to_save_np=path# путь для сохранений нумпай-массивов ПОМЕНЯЙТЕ НА СВОЙ ПУТЬ, КУДА ХОТИТЕ СОХРАНЯТЬ 10 ГБ!!!
########################################################################################################################
########################################################################################################################
inc_model=InceptionV3(include_top=False,
                      weights='imagenet',
                      input_shape=((3, image_size[0], image_size[1])))
bottleneck_datagen = ImageDataGenerator(rescale=1. / 255)  # собственно, генератор
train_generator = bottleneck_datagen.flow_from_directory(train_path,
                                                         target_size=image_size,
                                                         batch_size=batch_size,
                                                         class_mode=None,
                                                         shuffle=False)
validation_generator = bottleneck_datagen.flow_from_directory(validation_path,
                                                              target_size=image_size,
                                                              batch_size=batch_size,
                                                              class_mode=None,
                                                              shuffle=False)
bottleneck_features_train = inc_model.predict_generator(train_generator,steps=int(len(train_generator.filenames)/batch_size))# пока не разобрался со steps, мб для универсальности опустить этот параметр
np.save(open(path_to_save_np+'/bn_features_train.npy', 'wb'), bottleneck_features_train)
bottleneck_features_validation = inc_model.predict_generator(validation_generator,int(len(validation_generator.filenames)/batch_size))
np.save(open(path_to_save_np+'/bn_features_validation.npy', 'wb'), bottleneck_features_validation)
train_data = np.load(open(os.path.join(path_to_save_np,'bn_features_train.npy'), 'rb'))
train_labels = np.array([0] * int(train_data.shape[0]/2) + [1] * int(train_data.shape[0]/2))
validation_data = np.load(open(os.path.join(path_to_save_np,'bn_features_validation.npy'), 'rb'))
validation_labels = np.array([0] * int(validation_data.shape[0]/2) + [1] * int(validation_data.shape[0]/2)) # за счёт отсутсвия перемешивания(shuffle=False) в генераторе(flow_from_directory)
fc_model = Sequential()
fc_model.add(Flatten(input_shape=train_data.shape[1:]))
fc_model.add(Dense(64, activation='relu', name='dense_one'))
fc_model.add(Dropout(0.5, name='dropout_one'))
fc_model.add(Dense(64, activation='relu', name='dense_two'))
fc_model.add(Dropout(0.5, name='dropout_two'))
if n_classes==2:
    fc_model.add(Dense(1, activation='sigmoid', name='output'))
else:
    fc_model.add(Dense(n_classes, activation='softmax', name='output'))
fc_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
fc_model.fit(train_data, train_labels,
            nb_epoch=fc_nb_epoch, batch_size=batch_size,
            validation_data=(validation_data, validation_labels))
fc_model.save_weights(os.path.join(path_to_save_np,'fc_inception_cats_dogs_250.hdf5')) # сохраняем веса
fc_model.evaluate(validation_data, validation_labels)
################################################################################################
#                                           PART 2   UNITE 2 MODELS
weights_filename=os.path.join(path_to_save_np,'fc_inception_cats_dogs_250.hdf5')
x = Flatten()(inc_model.output)
x = Dense(64, activation='relu', name='dense_one')(x)
x = Dropout(0.5, name='dropout_one')(x)
x = Dense(64, activation='relu', name='dense_two')(x)
x = Dropout(0.5, name='dropout_two')(x)
if n_classes==2:
    top_model=Dense(1, activation='sigmoid', name='output')(x)
else:
    top_model = Dense(n_classes, activation='softmax', name='output')(x)
model = Model(input=inc_model.input, output=top_model)
model.load_weights(weights_filename, by_name=True) # загрузить веса в определённые слои по имени (by_name=True)
for layer in inc_model.layers[:inception_nontrainable_layers_count]:
    layer.trainable = False
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
                #optimizer='rmsprop',
              metrics=['accuracy']) # тонкая настройка (в первый раз использовали RMSProp, во второй раз используем стохастический градиентный бустинг для того, чтобы веса не слищком сильно обновлялись)
filepath=os.path.join(path_to_save_np,"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) # здесь происходит аугментация данных, в частности, горизонтальное отражение
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary')
pred_generator=test_datagen.flow_from_directory(validation_path,
                                                     target_size=image_size,
                                                     batch_size=batch_size2,
                                                     class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(train_data.shape[0]/batch_size),
        epochs=2,
        validation_data=validation_generator,
    validation_steps=np.ceil(validation_data.shape[0]/batch_size),
    callbacks=callbacks_list)
model.evaluate_generator(pred_generator, val_samples=batch_size2)# val_samples должен быть равен величине батча в генераторе!!!
imgs,labels=pred_generator.next() # загружает изображения в генератор и присваивает ей label
array_imgs=np.transpose(np.asarray([img_to_array(img) for img in imgs]),(0,2,1,3))
predictions=model.predict(imgs)
rounded_pred=np.asarray([np.round(i) for i in predictions])
print("Accuracy score: "+str(accuracy_score(labels,rounded_pred)))
print("Confusion matrix: ")
print(confusion_matrix(labels,rounded_pred))
print("F1-score(average='macro'): "+str(f1_score(labels, rounded_pred, average='macro')))
print("F1-score(average='micro'): "+str(f1_score(labels, rounded_pred, average='micro')))
print("F1-score(average='weighted'): "+str(f1_score(labels, rounded_pred, average='weighted')))
print("F1-score(average=None): ")
print(f1_score(labels, rounded_pred, average=None))
print("Precision-score(average='macro'):"+str(precision_score(labels, rounded_pred, average='macro')))
print("Precision-score(average='micro'): "+str(precision_score(labels, rounded_pred, average='micro')))
print("Precision-score(average='weighted'): "+str(precision_score(labels, rounded_pred, average='weighted')))
print("Precision-score(average=None): ")
print(precision_score(labels, rounded_pred, average=None))
print("Recall-score(average='macro'):"+str(recall_score(labels, rounded_pred, average='macro')))
print("Recall-score(average='micro'): "+str(recall_score(labels, rounded_pred, average='micro')))
print("Recall-score(average='weighted'): "+str(recall_score(labels, rounded_pred, average='weighted')))
print("Recall-score(average=None): ")
print(recall_score(labels, rounded_pred, average=None))
if n_classes==2:
    print("ROC_AUC score:" + str(roc_auc_score(labels, rounded_pred)))
    print("F1-score(average='binary'): " + str(f1_score(labels, rounded_pred, average='binary')))
    print("Precision-score(average='binary'):" + str(precision_score(labels, rounded_pred, average='binary')))
    print("Recall-score(average='binary'):" + str(recall_score(labels, rounded_pred, average='binary')))