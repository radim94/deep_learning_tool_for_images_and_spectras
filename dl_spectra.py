import pandas as pd
import pandas as pd
from keras.utils import to_categorical
#one-hot encode target column
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.layers import MaxPooling1D,GlobalAveragePooling1D,Dropout
from keras import optimizers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
#######################################################################################################################
#######################################################################################################################
#                                               ѕј–јћ≈“–џ ƒЋя »«ћ≈Ќ≈Ќ»я
path=r"C:\Users\Dmitry\output.xlsx"#r"C:\Users\16681613\Downloads\kozha_diagnoz_glazhennye_dlya_obedinenia.xlsx"
test_size=0.15## какую часть от всех данных отводить на тест
batch_size=8# размер батча
epochs=10# количество эпох
n_classes=6

########################################################################################################################
#                                                  ¬—ѕќћќ√ј“≈Ћ№Ќџ≈ ‘”Ќ ÷»»
def proba2class_num(y_pred):
    s=[]
    for arr in y_pred:
        s.append(np.argmax(arr))
    return np.array(s)
#########################################################################################################################
df=pd.read_excel(path)
df.head()
print(df.shape)
X=df.values[:,2:]
y=df.values[:,0]
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42,stratify=y)
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#здесь по-хорошему должен быть параметр stratify=y , чтобы в тесте и трейне содержание выборки
# было пропорционально исходной, но когда мало образцов класса может быть ошибка, что слищком мало образцов класса в трейне дл€ обучени€
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("Size of train set: "+str(X_train.shape))
print("Size of test set: "+str(X_test.shape))
print("Size of y_train: "+str(y_train.shape))
print("Size of y_test: "+str(y_test.shape))
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
model_m = Sequential()
#model_m.add(Reshape((334, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(16, 21, activation='relu', input_shape=(X.shape[1],1)))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(32, 11, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(64, 5, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Dense(2048, activation='softmax'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(32, 11, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Dense(2048, activation='softmax'))
model_m.add(Conv1D(32, 11, activation='relu'))
model_m.add(Conv1D(64, 5, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(350, activation='relu',kernel_initializer='lecun_uniform'))
model_m.add(Dropout(0.5))
model_m.add(Dense(100, activation='relu',kernel_initializer='lecun_uniform'))
model_m.add(Dropout(0.5))
model_m.add(Dense(350, activation='relu',kernel_initializer='lecun_uniform'))
model_m.add(Dropout(0.5))
model_m.add(Dense(100, activation='relu',kernel_initializer='lecun_uniform'))
model_m.add(Dropout(0.5))
model_m.add(Dense(n_classes, activation='softmax'))
print(model_m.summary())
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)#TODO заменить SGD optimizer на что-то другое
model_m.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model_m.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)
y_pred=model_m.predict(X_test)
y_test=proba2class_num(y_test)
y_pred=proba2class_num(y_pred)
print("Statistics for neural network:")
print("Accuracy score: "+str(accuracy_score(y_test,y_pred)))
print("Confusion matrix: ")
print(confusion_matrix(y_test,y_pred))
print("F1-score(average='macro'): "+str(f1_score(y_test,y_pred, average='macro')))
print("F1-score(average='micro'): "+str(f1_score(y_test,y_pred, average='micro')))
print("F1-score(average='weighted'): "+str(f1_score(y_test,y_pred, average='weighted')))
print("F1-score(average=None): ")
print(f1_score(y_test,y_pred, average=None))
print("Precision-score(average='macro'):"+str(precision_score(y_test,y_pred, average='macro')))
print("Precision-score(average='micro'): "+str(precision_score(y_test,y_pred, average='micro')))
print("Precision-score(average='weighted'): "+str(precision_score(y_test,y_pred, average='weighted')))
print("Precision-score(average=None): ")
print(precision_score(y_test,y_pred, average=None))
print("Recall-score(average='macro'):"+str(recall_score(y_test,y_pred, average='macro')))
print("Recall-score(average='micro'): "+str(recall_score(y_test,y_pred, average='micro')))
print("Recall-score(average='weighted'): "+str(recall_score(y_test,y_pred, average='weighted')))
print("Recall-score(average=None): ")
print(recall_score(y_test,y_pred, average=None))
if n_classes==2:
    print("ROC_AUC score:" + str(roc_auc_score(y_test, y_pred)))
    print("F1-score(average='binary'): " + str(f1_score(y_test, y_pred, average='binary')))
    print("Precision-score(average='binary'):" + str(precision_score(y_test, y_pred, average='binary')))
    print("Recall-score(average='binary'):" + str(recall_score(y_test, y_pred, average='binary')))
print("")
print("")
#####################################################################################################################
y=y.astype(int)
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42,stratify=y)
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
clf=GradientBoostingClassifier()
clf.fit(X_train,y_train)
y_pred_gb=clf.predict(X_test)
print("Statistics for gradient boosting:")
print("Accuracy score: "+str(accuracy_score(y_test,y_pred_gb)))
print("Confusion matrix: ")
print(confusion_matrix(y_test,y_pred_gb))
print("F1-score(average='macro'): "+str(f1_score(y_test,y_pred_gb, average='macro')))
print("F1-score(average='micro'): "+str(f1_score(y_test,y_pred_gb, average='micro')))
print("F1-score(average='weighted'): "+str(f1_score(y_test,y_pred_gb, average='weighted')))
print("F1-score(average=None): ")
print(f1_score(y_test,y_pred_gb, average=None))
print("Precision-score(average='macro'):"+str(precision_score(y_test,y_pred_gb, average='macro')))
print("Precision-score(average='micro'): "+str(precision_score(y_test,y_pred_gb, average='micro')))
print("Precision-score(average='weighted'): "+str(precision_score(y_test,y_pred_gb, average='weighted')))
print("Precision-score(average=None): ")
print(precision_score(y_test,y_pred_gb, average=None))
print("Recall-score(average='macro'):"+str(recall_score(y_test,y_pred_gb, average='macro')))
print("Recall-score(average='micro'): "+str(recall_score(y_test,y_pred_gb, average='micro')))
print("Recall-score(average='weighted'): "+str(recall_score(y_test,y_pred_gb, average='weighted')))
print("Recall-score(average=None): ")
print(recall_score(y_test,y_pred_gb, average=None))
if n_classes==2:
    print("ROC_AUC score:" + str(roc_auc_score(y_test, y_pred_gb)))
    print("F1-score(average='binary'): " + str(f1_score(y_test, y_pred_gb, average='binary')))
    print("Precision-score(average='binary'):" + str(precision_score(y_test, y_pred_gb, average='binary')))
    print("Recall-score(average='binary'):" + str(recall_score(y_test, y_pred_gb, average='binary')))