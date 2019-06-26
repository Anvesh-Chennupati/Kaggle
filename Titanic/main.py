import numpy as np
import pandas as pd
import seaborn as sns
# import pandas_profiling
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

if __name__ == '__main__':
    test_data = pd.read_csv("C:\\Users\\anves\\github\\Kaggle\\Titanic\\data\\test.csv")
    train_data = pd.read_csv("C:\\Users\\anves\\github\\Kaggle\\Titanic\\data\\train.csv")
    # profile = pandas_profiling.ProfileReport(test_data)
    # profile.to_file(outputfile="Titanic data profiling.html")

    # print(train_data.describe())
    # print(train_data.head(8))

    total = train_data.isnull().sum().sort_values(ascending=False)
    percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    # print(missing_data.head(5),sep = '\n')
    survived = 'survived'
    not_survived = 'not survived'

    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
    women = train_data[train_data['Sex']=='female']
    men = train_data[train_data['Sex']=='male']
    ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
    ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
    ax.legend()
    ax.set_title('Female')
    ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
    ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
    ax.legend()
    ax.set_title('Male')

    plt.show()

