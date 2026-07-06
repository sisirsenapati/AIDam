#Import Lib
import pandas as pd # Pandas
import numpy as np # numpy
import matplotlib.pyplot as plt # matplotlib pyplot
import seaborn as sns #seaborn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split # import for spilt training and test data.
from sklearn.preprocessing import StandardScaler # To scale the data using z-score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve,recall_score # Report and metrics
# Model Building
from sklearn.linear_model import LogisticRegression #Logistic
from sklearn.svm import SVC #SVC
from sklearn.tree import DecisionTreeClassifier #Decision
from sklearn.ensemble import RandomForestClassifier #RandomForest
#hyperparameter
# Metrics to evaluate the model
from sklearn import metrics
from sklearn import tree
# For tuning the model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
#ignore warning
import warnings
warnings.filterwarnings('ignore')