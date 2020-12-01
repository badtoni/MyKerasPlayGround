
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import pandas as pd
import os



def evaluate(history, folder_path):
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc      = history.history[     'accuracy' ]
    val_acc  = history.history[ 'val_accuracy' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]

    epochs   = range(len(acc)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     acc )
    plt.plot  ( epochs, val_acc )
    plt.title ('Training and validation accuracy')
    plt.xlabel('epochs [-]')
    plt.ylabel('accuracy [-]')

    plot_path = os.path.join(folder_path, 'accuracy_history.png')
    plt.savefig(plot_path)

    plt.figure()

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     loss )
    plt.plot  ( epochs, val_loss )
    plt.title ('Training and validation loss'   )
    plt.xlabel('epochs [-]')
    plt.ylabel('loss [-]')

    plot_path = os.path.join(folder_path, 'loss_history.png')
    plt.savefig(plot_path)

    plt.show()