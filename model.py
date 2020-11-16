import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers.experimental import preprocessing

    
def create_model(dataset,model_name,filepath):
    l = pd.read_csv(dataset)
    
    # first step always having a dataset which is labeled and setting up the model to be able to interpret those labels
    # this means you have to preprocess the data in some way - that depends on the inputs youre passing
    
    feature_names = ['product_issue','script_issue','testbed_issue']
    target_class = l['bug_type'].values
    bug_desc = l['bug_description'].values
    
    # from scikitlearn I've tried tfidf encoding, onehot encoding (dont do this for the input data you are training on, only labels)
    text_vectorizer = preprocessing.TextVectorization(output_mode="int")
    text_vectorizer.adapt(bug_desc)
    vocab_len = len(text_vectorizer.get_vocabulary()) + 2
    
    # onehot encoder changes labels into a 2d matrix which can be understood as a truth table, 
    # product, script, testbed becomes
    #[
    #    [p,mes]
    #    [t,mes]
    #    [s,mes]
    #]
    # => [ [1,0,0,mes] [0,1,0,mes],etc...] 
    # 1,0,0 when a given input is labeled as a product issue
    
    enc = OneHotEncoder()
    
    # you have to add the [:,np.newaxis] in order for the input shape to match what our model wants 
    # error you might see is something like shape (None,3) is not comapatible with shape (None,4) which is why I've added a new dimension to the matrix
    
    target_class = enc.fit_transform(np.array(target_class[:, np.newaxis])).toarray()
    
    # one thing to test later on is if lemmazation or other ways of preprocessing text so the model better UNDERSTANDS the text will increase accuracy
    
    
    # Size of vocabulary obtained when preprocessing text data, for now I have set this to 10k because of an issue I was getting
    # when it was the vocab length + 1
    num_words = 10000 
    # Number of classes for prediction outputs
    num_classes = 3  
    
    # this layer is the start of our model and says that we will accept a string input with shape (1,)
    # shape has to have the comma in order for the model to accept it - not 100% sure why
    
    description_input = keras.Input(shape=(1,),dtype="string", name="bug_description")
        
    # this layer is key to accepting raw strings as input as it will vectorize the input strings when theyre passed to /predict  
    # without text vectorization layer builtin to the model, a string like hello world worlds would be vectorized when sending your input as a request,
    # because of this your string wouldnt be processed with the model's vocabulary and would look something like this [0,1,2]
    # if you include the text vectorization layer in the model then it will interperet the string hello world worlds within its own vocabulary so it would look something more like [12, 26, 58] where these numbers represent some unique value within the model's vocabulary
    
    description_features = text_vectorizer(description_input)

    # the embedding layer creates word vectors out of inputs, passing the text vectorizer layer before this enables our model to use the words in context of our vocabulary
    description_features = layers.Embedding(input_dim=10000, output_dim=128)(description_features)

    # LSTM layer is what takes our word vectors and actually interprets the data heres a link - https://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
    description_features = layers.LSTM(64)(description_features)
    
    # this layer is responsible for removing some amount of overfitting from the model and should only be added in after LSTM
    # this is because the predictions the LSTM layer is creating can be overfit to our training data and dropout layer can solve this a bit
    # chaning the 0.2 gives different results, but I stopped at 0.2 because I got good results with this model 
    # its explained well here - https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
    description_features = layers.Dropout(0.2)(description_features)
    
    # finally we have a dense layer which actually will output the predictions
    # sigmoid activation is necessary (an alternative is softmax but youll notice that the model's accuracy never changes during training with it)
    # "sigmoid squash" is a commonly used term because this activation function takes the data and squashes them down to a number between 0-1 so it becomes a probability problem
    # whichever output has the highest value is also the bug type that an input is most likely to be!
    bug_class = layers.Dense(3, name="desc",activation='sigmoid')(description_features)

    # create the model object with 1 input layer and 1 output layer
    # can be modified to accept more as well!
    model = keras.Model(
        inputs=[description_input],
        outputs=[bug_class]#,infra_categorization]
    )
    # visualize the model architecture with this function
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    
     # split the data into training and testing sets 
    X_train,X_test,y_train,y_test = train_test_split(bug_desc,target_class,test_size=0.2,random_state=1)

    # compile the model with an optimizer which from my testing yielded the best results, and a loss function used for our type of problem 
    # use CC loss function when we have multiple labels and pass label data as a onehot encoded matrix
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    # finally fit the model on our data - a higher batch size can yield lower accuracies but also speeds up training
    # I had issues with the training taking a long time initially but that was before I changed preprocessing methods
    # ML requires A LOT of trial and error with tuning your parameters and finding a middle ground you are happy with
    # after just two epochs our training accuracy is almost 100% so 10 epochs is a sufficient # for this model
    # a batch size of 64 isn't too big or too small either and this combo yields good results
    model.fit(X_train,y_train,epochs=10,batch_size=64)
    
    save_path = os.path.join(filepath, f'{model_name}/1/')
    tf.saved_model.save(model, save_path)
