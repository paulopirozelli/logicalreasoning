import os
import sys

model_path = sys.argv[1]
dataset = sys.argv[2]
batch_size = sys.argv[3]

classifiers = ['1layer', '3layers']
layers = range(25)
learning_rates = ['1e-6', '1e-5']


for classifier in classifiers:
    for layer in layers:
        for learning_rate in learning_rates:
            os.system('python probing_layer.py ' + ' ' + model_path + ' ' + dataset + ' ' + 
                      classifier + ' ' + str(layer) + ' ' + learning_rate + ' ' + batch_size)