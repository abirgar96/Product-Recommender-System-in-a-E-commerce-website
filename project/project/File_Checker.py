from pathlib import Path
import os
def check_common_files_existence(intents, model, words, classes):
    if not(Path(model).is_file()) or not(Path(words).is_file()) or not(Path(classes).is_file()):
        print("missing some Model files")
        print("retraining the model")
        trainer_file = os.path.join(os.getcwd(), "project", "train_chatbot.py")
        command = "python " + trainer_file
        os.system(command)
    if not(Path(intents).is_file()):
        print ("missing intents.json")



