import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from pymongo import MongoClient
import nltk
from nltk.chat.util import Chat, reflections

# Download required NLTK data files
nltk.download('punkt')

# Step 1: Connect to MongoDB
client = MongoClient('localhost', 27017)  # Adjust the host and port if necessary
db = client['articles']  # Database name
collection = db['articles']  # Collection name

# Function to retrieve articles from MongoDB based on a keyword
def fetch_articles(keyword):
    articles = list(collection.find({"content": {"$regex": keyword, "$options": "i"}}))
    return articles

# Define patterns and responses
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, How are you today?", ]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there", ]
    ],
    [
        r"what is your name?",
        ["I am a chatbot created by you. You can call me ChatBot.", ]
    ],
    [
        r"how are you?",
        ["I'm doing good. How about you?", ]
    ],
    [
        r"sorry (.*)",
        ["It's alright", "It's OK, no problem", ]
    ],
    [
        r"I am fine",
        ["Great to hear that. How can I assist you today?", ]
    ],
    [
        r"(.*) age?",
        ["I am a computer program. I do not age.", ]
    ],
    [
        r"what (.*) want?",
        ["I want to help you with your queries.", ]
    ],
    [
        r"(.*) created you?",
        ["I was created by a brilliant programmer.", ]
    ],
    [
        r"(.*) (location|city) ?",
        ["I am in the cloud.", ]
    ],
    [
        r"search for (.*)",
        ["Let me find articles related to %1 for you.", ]
    ],
    [
        r"quit",
        ["Bye, take care. See you soon.", "It was nice talking to you. Goodbye!"]
    ],
    [
        r"(.*)",
        ["I am sorry, I do not understand. Can you please rephrase?", ]
    ],
]

def chatbot_response(user_input):
    for pattern, responses in pairs:
        match = nltk.re.match(pattern, user_input)
        if match:
            response = responses[0]
            if '%1' in response:
                keyword = match.group(1)
                articles = fetch_articles(keyword)
                if articles:
                    response = "I found the following articles related to '{}':\n".format(keyword)
                    for article in articles:
                        response += "\nTitle: {}\nContent: {}\n".format(article['title'], article['content'])
                else:
                    response = "I couldn't find any articles related to '{}'.".format(keyword)
            return response
    return "I am sorry, I do not understand. Can you please rephrase?"

# Function to interact with the chatbot
def chatbot():
    print("Hi, I am a chatbot created by you. Ask me anything or type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("ChatBot: Bye, take care. See you soon.")
            break
        response = chatbot_response(user_input)
        print("ChatBot:", response)

if __name__ == "__main__":
    chatbot()
