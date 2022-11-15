import random
import json
import torch
import models
import utils

class Chatbot:

    FILE = "data.pth"

    with open('./assets/intents.json', 'r') as f:
        intents = json.load(f)

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = torch.load(self.FILE)
        self.bot_name = "ChickenBenny"
        self.prep = utils.Preprocessor()
        self._get_info()
        self._make_model()

    def _get_info(self):
        self.input_size = self.data["input_size"]
        self.hidden_size = self.data["hidden_size"]
        self.output_size = self.data["output_size"]
        self.all_words = self.data['all_words']
        self.tags = self.data['tags']
        self.model_state = self.data["model_state"]

    def _make_model(self):
        self.model = models.ANN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

    def chat(self):
        print("Let's chat! (type 'quit' to exit)")
        while True:
            sentence = input("You: ")
            if sentence == "quit":
                break

            sentence = self.prep.chatbot_msg_process(sentence)
            x = self.prep.bag_of_words(sentence, self.all_words)
            x = x.reshape(1, x.shape[0])
            x = torch.from_numpy(x).to(self.device)

            output = self.model(x)
            _, predicted = torch.max(output, dim=1)

            tag = self.tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in self.intents['intents']:
                    if tag == intent["tag"]:
                        print(f"{self.bot_name}: {random.choice(intent['responses'])}")
            else:
                print(f"{self.bot_name}: I do not understand...")


if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot.chat()