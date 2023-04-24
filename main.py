import pyttsx3, datetime, time
import speech_recognition as sr
engine = pyttsx3.init()

NAME = "DAVID"
##davids age==========

DAVIDS_AGE = 3

import nltk, os,function2
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()# used to stem our words
import numpy
import random
import json, pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

####============================preprocess our data=============================


with open('intents.json')as file:
    data = json.load(file)
if 'data.pickle' in os.listdir(os.getcwd()):    ##load some pickle data
    with open('data.pickle','rb') as f:
        words,labels,training, output =pickle.load(f)
else:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            ##== stemming the words in the pattern i.e remove all symbols dat might make the word look different
            ## to  get and stem the words we need to tokenize the statements
            ## tokenize just gets all the words in out pattern array
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])#tags for each word
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    words = [stemmer.stem(w.lower())for w in words if w !="?"]## stemming the words
    words = sorted(list(set(words)))#removes all duplicate elements
    labels = sorted(labels)#remove the duplicates in labels list

    ## creating a bag of words for all our words

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
    #print(out_empty,"=========",len(labels))
    #print(labels)
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        #print(wrds)
        #print(words)
        for w in words:
            if w in wrds:
                #print(w)
                bag.append(1)
            else:
                bag.append(0)
        output_row = list(out_empty)
        output_row[labels.index(docs_y[x])]=1
        training.append(bag)
        output.append(output_row)

        print(output_row, labels)
    #print(training,'\n',output)
    #####======converting the trainning and output data in numpy arrays
    training = numpy.array(training)##training data=========
    output = numpy.array(output)

    ##save to pickle file
    with open('data.pickle','wb') as f:
        pickle.dump((words,labels,training, output),f)

#########################creating the model===========
model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]),activation='softmax'))
sgd = SGD(lr =0.01,decay=1e-6,momentum=0.9, )
model.compile(loss="categorical_crossentropy",optimizer=sgd, metrics=['accuracy'])
if 'chatbotmodel.h5' in os.listdir(os.getcwd()):
    model= load_model('chatbotmodel.h5')
else:
    hist = model.fit(numpy.array(training),numpy.array(output),epochs=1000,batch_size=8,
                     verbose=1)
    model.save('chatbotmodel.h5', hist)
    print('DONE====================')

def bag_of_words(sentence,words):
    bag = [0 for _ in range(len(words))]
    s_words =nltk.word_tokenize(sentence)
    s_words =[stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w ==se:
                bag[i]=1
    return numpy.array(bag)

'''def find_out(s):
    if "who" in s or "what" in s or"when"'''
def chat():
    print("start talking with the bot!")
    while True:
        inp = input("You:  ")
        #print(inp, "===============================")
        if inp.lower() == "quit":
            break
        if "play" in inp:
            index=inp[inp.index("play")+4:]
            function2.get_video(index)
        if "search" in inp or "find" in inp:
            index = inp[inp.index("play") + 5:]
            function2.search(index)
        else:
            results = model.predict(numpy.array([bag_of_words(inp,words)]))
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            ans = random.choice(responses)
            if "{NAME}" in ans:
                ans = ans.replace("{NAME}", str(NAME))
            if "NAME" in ans:
                ans = ans.replace("NAME", str(NAME))
            if '{AGE}' in ans:
                ans = ans.replace('{AGE}', str(DAVIDS_AGE))
            if "AGE" in ans:
                ans = ans.replace('AGE', str(DAVIDS_AGE))

            #print(ans)
            engine.say(ans)
            engine.runAndWait()


chat()