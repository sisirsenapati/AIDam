from common import *
from common.helper import run_all_models_dashboard

# import common as com


# load dataset from excel file
df = pd.read_excel("data\HR_Employee_Attrition_Dataset.xlsx")

excludecols = df.columns[(df.nunique() == len(df)) | (df.nunique() == 1)]
df = df.drop(columns=excludecols,axis=1) 
# Numaric cols and Category cols identification
num_cols = df.select_dtypes(include=np.number).columns #include='number'
cat_cols = df.select_dtypes(include='object').columns
# com.describe_insights(df, num_cols) # Call the function to describe insights for numerical columns
corr = df[num_cols].corr().round(2)
dummy_cat_col = [col for col in df.select_dtypes(include='object').columns if col != 'Attrition']

df_backup = df.copy()    

df = pd.get_dummies(data=df,columns=dummy_cat_col,drop_first=True)
dict_att = {'Yes':1, 'No':0}
df['Attrition'] = df['Attrition'].map(dict_att)

# Separating the independent variables (X) Only Attrition and the dependent variable (Y) Except Attrition
Y = df['Attrition']
X = df.drop(columns=['Attrition'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1, stratify = Y)

# Scaling the data make the data evenly make the unform watege based to make the numbers uniform
sc = StandardScaler()
X_train_scaled=sc.fit_transform(X_train) 
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)  
X_test_scaled = sc.transform(X_test) 

############################################ Logistic Regression Model ############################
lg = LogisticRegression()
lg.fit(X_train_scaled,y_train)
y_pred_train = lg.predict(X_train_scaled)
y_scores_lg = lg.predict_proba(X_train_scaled)
precisions_lg, recalls_lg, thresholds_lg = precision_recall_curve(y_train, y_scores_lg[:, 1])

# Find the intersection point (where precision == recall)
intersection_idx = np.argmin(np.abs(precisions_lg - recalls_lg))
intersection_threshold = thresholds_lg[intersection_idx]

# Improve Permformance
optimal_threshold = intersection_threshold.round(2)
# metrics_score(y_train, y_pred_train[:,1]>optimal_threshold)

# optimal_threshold=.35
y_pred_train = lg.predict_proba(X_train_scaled)

# com.metrics_score1(y_train, y_pred_train[:,1]>optimal_threshold)

############################## Support Vector Machines #####################################
######################### Linear Kernel ###################
svmlnr = SVC(kernel='linear')
modellnr = svmlnr.fit(X = X_train_scaled, y = y_train)
y_pred_train_svmlnr = modellnr.predict(X_train_scaled)
# com.metrics_score1(y_train,y_pred_train_svmlnr)
y_pred_test_svm = modellnr.predict(X_test_scaled)
# com.metrics_score1(y_test, y_pred_test_svm)
 ############################# RBF Kernel ########################
svm_rbf=SVC(kernel='rbf',probability=True)
svm_rbf.fit(X_train_scaled,y_train)
optimal_threshold_svm=.35
y_pred_train = svm_rbf.predict_proba(X_train_scaled)
# com.metrics_score1(y_train, y_pred_train[:,1]>optimal_threshold_svm)

## Compute permutation feature importance
# result = permutation_importance(svm_rbf, X_test_scaled, y_test, n_repeats=30, random_state=42)

## Extract importance and arrange features in order
# feature_importances = result.importances_mean
# sorted_idx = np.argsort(feature_importances)

# Plot feature importance
# plt.figure(figsize=(13, 13))
# plt.barh(range(X_test_scaled.shape[1]), feature_importances[sorted_idx], align='center')
# plt.yticks(range(X_test_scaled.shape[1]), np.array(X.columns)[sorted_idx])
# plt.xlabel("Permutation Feature Importance")
# plt.title("Feature Importance for SVM with RBF Kernel")
# plt.show()
############################ Decision Tree ######################################
dt = DecisionTreeClassifier(class_weight = {0: 0.84, 1: 0.16}, random_state = 1)
dt.fit(X_train, y_train)
y_train_pred_dt = dt.predict(X_train)
# com.metrics_score1(y_train, y_train_pred_dt)
## 
# importances = dt.feature_importances_
# columns = X.columns
# importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)
# plt.figure(figsize = (13, 13))
# sns.barplot(x=importance_df.Importance, y=importance_df.index)

#################### Random Forest ##########################
rf_estimator = RandomForestClassifier(class_weight = {0: 0.84, 1: 0.17}, random_state = 42)
rf_estimator.fit(X_train, y_train)
y_pred_train_rf = rf_estimator.predict(X_train)
# com.metrics_score1(y_train, y_pred_train_rf)
##

# importances = rf_estimator.feature_importances_
# columns = X.columns
# importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)
# plt.figure(figsize = (13, 13))
# sns.barplot(x=importance_df.Importance, y=importance_df.index) 

########################## Hyper Parameter Tuning for Decision Tree ##########################
# dtree_estimator = DecisionTreeClassifier(class_weight = {0: 0.17, 1: 0.83}, random_state = 1)
# parameters = {'max_depth': np.arange(2, 100),
#               'criterion': ['gini', 'entropy'],
#               'min_samples_leaf': [5, 10, 20, 25]
#              }
# scorer = metrics.make_scorer(recall_score, pos_label = 1)
# gridCV = GridSearchCV(dtree_estimator, parameters, scoring = scorer, cv = 10)
# gridCV = gridCV.fit(X_train, y_train)
# dtree_estimator = gridCV.best_estimator_
# dtree_estimator.fit(X_train, y_train)
# y_train_pred_dt = dtree_estimator.predict(X_train)
# com.metrics_score1(y_train, y_train_pred_dt)

##
# importances = dtree_estimator.feature_importances_
# columns = X.columns
# importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)
# plt.figure(figsize = (13, 13))
# sns.barplot(x=importance_df.Importance, y=importance_df.index)

########################## Hyper Parameter Tuning for Random Forest ##########################
# rf_estimator_tuned = RandomForestClassifier(class_weight = {0: 0.17, 1: 0.83}, random_state = 1)
# params_rf = {
#         "n_estimators": [70, 35,50],
#         "min_samples_leaf": np.arange(1, 7, 1),
#         "max_features": [0.7,0.9,0.8,0.5,0.6, 'auto'],
# }
# scorer = metrics.make_scorer(recall_score, pos_label = 1)
# grid_obj = GridSearchCV(rf_estimator_tuned, params_rf, scoring = scorer, cv = 5)
# grid_obj = grid_obj.fit(X_train, y_train)
# rf_estimator_tuned = grid_obj.best_estimator_
# rf_estimator_tuned.fit(X_train, y_train)
# y_train_pred_rf = rf_estimator_tuned.predict(X_train)
# com.metrics_score1(y_train, y_train_pred_rf)

# importances = rf_estimator_tuned.feature_importances_
# columns = X.columns
# importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)
# plt.figure(figsize = (13, 13))
# sns.barplot(x=importance_df.Importance, y=importance_df.index)

###########################  All Model one place #######################################
run_all_models_dashboard(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled
)