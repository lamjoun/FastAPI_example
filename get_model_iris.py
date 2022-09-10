import joblib
import random
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def print_head_and_tail(i_df):
    print(i_df.head())
    print(i_df.tail())


def get_random_values(i_df, i_nbr_of_elm):
    if (i_nbr_of_elm < len(i_df)) and (i_nbr_of_elm > 0):
        l_list = i_df.index.to_list()
        random.shuffle(l_list)
        l_df = i_df.iloc[l_list[0:i_nbr_of_elm-1]]
        return l_df


ds_iris = datasets.load_iris()
#
print("\niris.feature_names:\n", ds_iris.feature_names, "\n\n")
print("iris.target_names:\n", ds_iris.target_names, "\n\n")
#
X_array = ds_iris.data
y_array = ds_iris.target
#
print("X.shape=", X_array.shape)
print("Y.shape=", X_array.shape)
#
print("Check target values=", set(y_array[:, ]))

df = pd.DataFrame(ds_iris.data, columns=ds_iris.feature_names)
df["target"] = ds_iris["target"]
print_head_and_tail(df)
#
X = df.drop(['target'], axis=1)
y = df['target']
print_head_and_tail(X)
print_head_and_tail(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=df['target'], random_state=1)

model = DecisionTreeClassifier(max_depth=5, random_state=1)
model.fit(X_train, y_train)
#
joblib.dump(model, './model/model.sav')
#
loaded_model = joblib.load('./model/model.sav')
#
y_pred = loaded_model.predict(X_test)
print(classification_report(y_test, y_pred))

df_random = get_random_values(df, 10)
print()
df_random.to_csv('random_values_from_dataset.csv', index=False)

input_list = [3, 5, 7, 9]
#
#
df1 = pd.DataFrame([input_list],
                  columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

print(df1.head())




