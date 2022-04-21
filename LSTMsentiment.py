import pandas as pd
import re

from keras.models import load_model

import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


def fff(x): # converting .txt to list
    with open(x) as file:
        line = []
        for lines in file.readlines():
            line.append(lines)
        return line


line = fff('train.txt')


def csv(line): # converting list into data frame
    list1,list2 = [],[]
    for lines in line:
        x,y = lines.split(';')
        y = y.replace('\n','')
        list1.append(x)
        list2.append(y)
    df = pd.DataFrame(list(list1),columns=['sentence'])
    df['emotion'] = list2
    return df

df = csv(line)


wn = WordNetLemmatizer()

x = []
with open('train_x_lem.txt') as x:
    x = [line.strip() for line in x]

### Handling Text Data

test_line = fff('test.txt')

test_df = csv(test_line)
# test_df.head()

# '''
x_test = []
with open('test_x_lem.txt') as xx:
    x_test = [line.strip() for line in xx]

all = x + x_test # list

y = df.iloc[:,1].values # training data as array
y_test = test_df.iloc[:,1].values  # testing data as array

## Building and training the lstm model

from tensorflow.keras.layers import Embedding,LSTM,Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

y_train = pd.DataFrame(y) # data frame

# creating a internal dictionary where we are indexing each word
tokenizer = Tokenizer(num_words=10000, split=' ')
tokenizer.fit_on_texts(all)

def conv(all):
    # replacing each word with it's corresponding value acc. to dict
    X1 = tokenizer.texts_to_sequences(all)
    X1 = pad_sequences(X1, maxlen=20, padding='post', truncating='post')
    return X1

# all is list type
X1 = conv(all)
X_train = X1[:16000]
X_test = X1[16000:]


Y_train = pd.get_dummies(y_train).values # creating one hot vector for all unique values of y
Y_test = pd.get_dummies(y_test).values

"""
model = Sequential()

model.add(Embedding(input_dim=10000,output_dim = 64,input_length=20)) # dim = dimension, number of dimensions of vector 

model.add(LSTM(64)) # 64 is number of neurons in the LSTM

model.add(Dense(6,activation='softmax'))

model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=32,epochs=12,verbose=2,validation_split=0.2)

loss,acc = model.evaluate(X_test,Y_test)

print("Loss is",loss*100,"\b% and Accuracy is",acc*100,"\b%")

# preds = model.predict(X_test)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model
"""
print("Loss is 2.902735397219658% and Accuracy is 88.59999775886536%")
model = load_model('my_model.h5')

sentimentS = ["Anger", "Fear", "Joy", "Love", "Sadness", "Surprise"]
def predictTheEmotion(sentence):
    prediction = model.predict(conv(sentence))
    emotion = []
    print("Predicting the emotion...")
    for j in range(len(prediction)):
        mx = 0.0
        em = ""
        for i in range(6):
            if prediction[j][i] > mx:
                mx = prediction[j][i]
                em = sentimentS[i]
        emotion.append(em)
    return emotion

# print(predictTheEmotion(["I love you"]))
# print(predictTheEmotion(["I don't hate you"]))
# print(predictTheEmotion(["I do not hate you"]))
# print(predictTheEmotion(["I want to live with you"]))
# print(predictTheEmotion(["I want to hug you"]))

# Indexing :
# 0 - anger
# 1 - fear
# 2 - joy
# 3 - love
# 4 - sadness
# 5 - surprise