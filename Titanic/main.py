import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import style

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

test_data = pd.read_csv("C:\\Users\\anves\\github\\Kaggle\\Titanic\\data\\test.csv")
train_data = pd.read_csv("C:\\Users\\anves\\github\\Kaggle\\Titanic\\data\\train.csv")

print(train_data.describe())

