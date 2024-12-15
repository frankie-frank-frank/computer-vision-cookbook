'''
This project is a gentle implementation of linear regression while training a car price prediction model
It takes a (N,8) input data used to predict a (N, 1) output to predict next output
'''

import tensorflow as tf
import pandas as pd
import seaborn as sns

data = pd.read_csv('train.csv', ",")