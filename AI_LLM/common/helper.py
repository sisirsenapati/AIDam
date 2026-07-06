#Import Lib
import pandas as pd # Pandas
import numpy as np # numpy
import matplotlib.pyplot as plt # matplotlib pyplot
import seaborn as sns #seaborn
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


def describe_insights(df, cols):
    summary = df[cols].describe().T

    for col in summary.index:
        insights = []

        if summary.loc[col, 'count'] < len(df):
            insights.append("Missing values") #It checks if the number of filled rows (count) is less than the total rows
                                               # in the entire table (len(df)). If true, it means some values are completely blank.

        if summary.loc[col, 'std'] == 0:
            insights.append("Constant") #The std (standard deviation) measures how much the numbers change.
            # If it equals 0, the numbers never change at all (e.g., every single row has the exact same value).
  #  Skewness -- Asymmetry

        if summary.loc[col, 'mean'] > summary.loc[col, '50%']:
            insights.append("Right skewed")
        elif summary.loc[col, 'mean'] < summary.loc[col, '50%']:
            insights.append("Left skewed")
# What it does: It compares the average (mean) against the middle value (50% median).If the average is dragged higher than the middle,
# it is Right skewed (a few huge numbers are pulling the average up).If the average is dragged lower,
# it is Left skewed (a few tiny numbers are pulling the average down).
        if summary.loc[col, 'max'] > summary.loc[col, '75%'] * 1.5:
            insights.append("High outliers")

        elif summary.loc[col, 'min'] < summary.loc[col, '25%'] * 1.5:
            insights.append("Low outliers")
#  Outliers (Extreme Values)
# What it does: It looks for numbers that stand too far away from the rest of the pack.
# If the max value is exceptionally higher than the 75th percentile mark, it flags High outliers.
# If the min value is exceptionally lower than the 25th percentile mark, it flags Low outliers.
#  (Note: There is a small typo in the low outlier formula in this specific code block—it multiplies instead of dividing or subtracting,
#   but the logical intent is to scan for abnormally small numbers).
        print(f"{col}: {', '.join(insights)}")
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))

    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
def metrics_score1(actual,prediction):
  cr = classification_report(actual,prediction)
  cm = confusion_matrix(actual, prediction)

  print('Classification report : \n',cr)
  print('*'*50)
  print('Confusion matrix : \n',cm)
  print('*'*50)
  plt.figure(figsize=(8,5))
  sns.heatmap(cm,annot=True,fmt='0.2f',xticklabels=['Non-Attries','Attries'], yticklabels=['Non-Attries','Attries'])
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.show()            
def logit2prob(logr,x):
  log_odds = logr.coef_ * x + logr.intercept_
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

# -----------------------------
# Your Metrics Function (Modified for subplot support)
# -----------------------------
def metrics_score11(actual, prediction, ax, title):
    cm = confusion_matrix(actual, prediction)

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Non-Attrition','Attrition'],
                yticklabels=['Non-Attrition','Attrition'],
                ax=ax)

    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
# -----------------------------
# MAIN GENERIC FUNCTION
# -----------------------------
def run_all_models_dashboard(X_train, X_test, y_train, y_test, X_train_scaled):

    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    axes = axes.flatten()

    plot_idx = 0

    # ============================
    # 1. Logistic Regression
    # ============================
    lg = LogisticRegression()
    lg.fit(X_train_scaled, y_train)

    y_pred = lg.predict(X_train_scaled)
    metrics_score11(y_train, y_pred, axes[plot_idx], "Logistic Default")
    plot_idx += 1

    # Threshold tuning
    y_scores = lg.predict_proba(X_train_scaled)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

    # Find intersection
    intersection_idx = np.argmin(np.abs(precisions - recalls))
    optimal_threshold = thresholds[intersection_idx].round(2)

    y_pred_opt = (y_scores > optimal_threshold)
    metrics_score11(y_train, y_pred_opt, axes[plot_idx],
                   f"Logistic Tuned (thr={optimal_threshold})")
    plot_idx += 1


    # ============================
    # 2. SVM Linear
    # ============================
    svm_lin = SVC(kernel='linear')
    svm_lin.fit(X_train_scaled, y_train)

    y_pred = svm_lin.predict(X_train_scaled)
    metrics_score11(y_train, y_pred, axes[plot_idx], "SVM Linear")
    plot_idx += 1


    # ============================
    # 3. SVM RBF
    # ============================
    svm_rbf = SVC(kernel='rbf', probability=True)
    svm_rbf.fit(X_train_scaled, y_train)

    y_scores = svm_rbf.predict_proba(X_train_scaled)[:,1]
    y_pred = (y_scores > 0.35)

    metrics_score11(y_train, y_pred, axes[plot_idx], "SVM RBF (thr=0.35)")
    plot_idx += 1


    # ============================
    # 4. Decision Tree
    # ============================
    dt = DecisionTreeClassifier(class_weight={0:0.84,1:0.16}, random_state=1)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_train)
    metrics_score11(y_train, y_pred, axes[plot_idx], "Decision Tree")
    plot_idx += 1


    # ============================
    # 5. Random Forest
    # ============================
    rf = RandomForestClassifier(class_weight={0:0.84,1:0.17}, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_train)
    metrics_score11(y_train, y_pred, axes[plot_idx], "Random Forest")
    plot_idx += 1


    # ============================
    # 6. Decision Tree (Tuned)
    # ============================
    dt_tune = DecisionTreeClassifier(class_weight={0:0.17,1:0.83}, random_state=1)

    params = {
        'max_depth': np.arange(2, 20),
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [5,10,20]
    }

    scorer = metrics.make_scorer(recall_score, pos_label=1)

    grid = GridSearchCV(dt_tune, params, scoring=scorer, cv=5)
    grid.fit(X_train, y_train)

    best_dt = grid.best_estimator_
    y_pred = best_dt.predict(X_test)

    metrics_score11(y_test, y_pred, axes[plot_idx], "DT Tuned (Test)")
    plot_idx += 1


    # ============================
    # 7. Random Forest (Tuned)
    # ============================
    rf_tune = RandomForestClassifier(class_weight={0:0.17,1:0.83}, random_state=1)

    params_rf = {
        "n_estimators": [50,70],
        "min_samples_leaf": np.arange(1,5),
        "max_features": [0.7,0.8,0.9]
    }

    grid = GridSearchCV(rf_tune, params_rf, scoring=scorer, cv=5)
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    y_pred = best_rf.predict(X_test)

    metrics_score11(y_test, y_pred, axes[plot_idx], "RF Tuned (Test)")
    plot_idx += 1


    # Remove unused plots
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()      