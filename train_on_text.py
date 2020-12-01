import os
import sys
from models.model_sentiment0.transfer_lstm_sentiment_anal import train_network




def start_training(folder_path):

    # TODO insert some control functions

    train_network(folder_path)





if __name__ == '__main__':
    if len(sys.argv) > 1:
        # print(arg)
        folder_path = sys.argv[1]
        start_training(folder_path)
    else:

        folder_path = "models/model_sentiment1"

        start_training(folder_path)