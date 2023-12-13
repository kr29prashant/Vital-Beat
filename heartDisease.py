import numpy as np
import pandas as pd
import sklearn as sk
# print(pd.__version__)
# print(sk.__version__)


data = pd.read_csv('heart.csv')


data.isnull().sum()

data_dup = data.duplicated().any()


data = data.drop_duplicates()

dataset = data
cate_val = []
cont_val = []

for column in data.columns:
  if data[column].nunique() <= 10:
    cate_val.append(column)
  else:
    cont_val.append(column)

# print(cate_val)
data['cp'].unique()

cate_val.remove('sex')
cate_val.remove('target')
#
# data = pd.get_dummies(data, columns = cate_val, drop_first=True)
#
# print(data.head())
#
from sklearn.preprocessing import StandardScaler
#
st = StandardScaler()
data[cate_val] = st.fit_transform(data[cate_val])
data[cont_val] = st.fit_transform(data[cont_val])

print(data.head())


X = data.drop('target',axis=1)
Y = data['target']

# print(data.head())




from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# rf = RandomForestClassifier(n_estimators = 150, oob_score = True, n_jobs = -1,random_state =100, min_samples_leaf = 50)


rf = RandomForestClassifier()
rf.fit(X_train,Y_train)

X_test_prediction6 = rf.predict(X_test)
test_data_accuracy6 = accuracy_score(Y_test,X_test_prediction6)
print(test_data_accuracy6)


# inputdata = [51.0, 2.0, 100.0, 1.0, 222.0, 0.0, 1.0, 143.0, 1.0, 1.2, 1.0, 0.0, 2.0]
new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1,
    'slope':2,
    'ca':2,
    'thal':3,
},index=[0])

# st = StandardScaler()
# new_data[cont_val] = st.fit_transform(new_data[cont_val])


print(new_data)


# X = dataset.drop('target',axis=1)
# Y = dataset['target']
#
# from sklearn.ensemble import RandomForestClassifier
#
# rf = RandomForestClassifier()
#
# rf.fit(X,Y)

pred = rf.predict(new_data)
#
print(pred)

# print(X.shape)

# print(new_data.head())
#
#
# new_data = pd.get_dummies(new_data, columns = cate_val, drop_first=True)
#
# print(new_data.head())


# inputDataAsNumpy = np.asarray(inputdata)
#
# inputdataReshaped = inputDataAsNumpy.reshape(1,-1)

# prediction = rf.predict(new_data)
# print(new_data)
# print(prediction)

# import pickle
#
# filename = "heartRandomForest.pkl"
#
# # save model
# pickle.dump(rf, open(filename, "wb"))
#
# # load model
# loaded_model = pickle.load(open(filename, "rb"))































