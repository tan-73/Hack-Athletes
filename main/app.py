import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

#Imports
import nltk
import json
import random
import pickle as p
import numpy as np

from nltk.stem import WordNetLemmatizer
from keras.src.saving.saving_api import load_model
from tkinter import *

class App():
    def __init__(self):
        """
        Initialization
        """

        self.lemmatizer = WordNetLemmatizer()
        self.model = load_model("main\\chatbot_model.h5")

        self.intents = json.loads(open("intents.json").read())
        self.words = p.load(open(r"main\words.pkl", "rb"))
        self.tags = p.load(open(r"main\\tags.pkl", "rb"))
        ...
    
    def clean_up(self, sentence):
        """
        Tokenization and Lemmatization

        :param sentence: sentence
        :type sentence: string
        """
        return [self.lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]

    def BoW(self, sentence, words, details = True):
        """
        returns a Bag of Words

        :param sentence: sentence
        :type sentence: string
        :param word: word
        :type word: string
        :param details: show the details, defaults to True
        :type details: bool, optional
        """
        sentence_words = self.clean_up(sentence)
        bag = [0] * len(words)
        for sentence in sentence_words:
            for index, word in enumerate(words):
                if word == sentence:
                    bag[index] = 1
                    if details:
                        print(f"found: {word}")
        return (np.array(bag))

    def predict_tag(self, sentence, model):
        """prediction of model

        :param sentence: sentence
        :type sentence: string
        :param model: model
        :type model: model
        :return: list
        :rtype: list
        """
        p = self.BoW(sentence, self.words, details = False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key = lambda x: x[1], reverse = True)
        return_list = []

        for r in results:
            return_list.append({"intent": self.tags[r[0]], "probability": str(r[1])})
        if not return_list:
            return_list.append({"intent": "no_match", "probability": "0"})
        return return_list
    
    def get_response(self, ints, intents):
        """creates response

        :param ints: intent
        :type ints: string
        :param intents: intents
        :type intents: list
        :return: list
        :rtype: list
        """
        t = ints[0]["intent"]
        if t == "no_match":
            return "I'm sorry, I don't understand that."
        list_of_intents = intents["intents"]
        for i in list_of_intents:
            if(i["tag"] == t):
                result = random.choice(i["responses"])
                break
        return result
    
    def bot_response(self, text):
        """return response

        :param text: test
        :type text: string
        :return: text
        :rtype: string
        """
        ints = self.predict_tag(text, self.model)
        res = self.get_response(ints, self.intents)
        return res
    
app = App()

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = app.bot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5, bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff', command= send )
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")

def on_enter(e):
    send()
base.bind('<Return>', on_enter)
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()













