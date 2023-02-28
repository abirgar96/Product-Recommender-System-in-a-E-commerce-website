import os 


intents_file=os.path.join(os.getcwd(), "Model", "intents.json")
model_file=os.path.join(os.getcwd(),"Model", "chatbot_model.h5")
words_file=os.path.join(os.getcwd(),"Model", "words.pkl")
classes_file=os.path.join(os.getcwd(), "Model", "classes.pkl")

print(intents_file)