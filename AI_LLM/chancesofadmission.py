from common import *
from common.helper import *

df = pd.read_csv("data\Admission_Predict.csv")
# print(df.head())
excludecols = df.columns[(df.nunique() == len(df)) | (df.nunique() == 1)]
# print(excludecols)
df = df.drop(columns=excludecols,axis=1) 

# print(df.nunique())

df["Chance of Admit "] = df["Chance of Admit "].apply(lambda x: 1 if x >= 0.8 else 0)

# print(df["Chance of Admit "].value_counts())

# for i in df.columns:
#     print(df[i].value_counts(normalize=True))
#     print('*'*40)
dfcol = ["University Rating"]
df = pd.get_dummies(data=df,columns=dfcol,drop_first=True)

Y = df['Chance of Admit ']
X = df.drop(columns=['Chance of Admit '])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1, stratify = Y)

sc = StandardScaler()
X_train_scaled=sc.fit_transform(X_train) 
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)  
X_test_scaled = sc.transform(X_test) 

# run_all_models_dashboard(
#     X_train,
#     X_test,
#     y_train,
#     y_test,
#     X_train_scaled
# )
# print(df.describe().T)
# plt.figure(figsize=(15,8))
# sns.scatterplot(data=df,
#            x='GRE Score',
#            y='TOEFL Score',
#            hue='Chance of Admit ',
#            size='SOP');
# plt.show()

plt.figure(figsize=(10,7))
sns.boxplot(data=df,
             x='LOR ',
             y='GRE Score',
             hue='Chance of Admit ')
plt.title('Relationship between different LOR and GRE Score')
plt.show()