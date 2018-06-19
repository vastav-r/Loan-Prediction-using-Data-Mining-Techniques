import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv(r'dataset/train.csv',header=0)
#conversions
d={'Male':1,'Female': 0}
df['Gender']=df['Gender'].map(d)
d={'Y':1, 'N':0}
df['Married']=df['Married'].map(d)
df['Education']=df['Education'].map(d)
df['Self_Employed']=df['Self_Employed'].map(d)
df['Loan_Status']=df['Loan_Status'].map(d)
d={'Urban':2,'Semiurban':1,'Rural':0}
df['Property_Area']=df['Property_Area'].map(d)

#selection
features=list(df.columns[1:12])

y=df["Loan_Status"]
x=df[features]
clf=RandomForestClassifier(n_estimators=10)
clf=clf.fit(x,y)

df=pd.read_csv(r'dataset/test.csv',header=0)
d={'Male':1,'Female': 0}
df['Gender']=df['Gender'].map(d)
d={'Y':1, 'N':0}
df['Married']=df['Married'].map(d)
df['Education']=df['Education'].map(d)
df['Self_Employed']=df['Self_Employed'].map(d)
d={'Urban':2,'Semiurban':1,'Rural':0}
df['Property_Area']=df['Property_Area'].map(d)

#prediction
features=list(df.columns[1:12])

test_x=df[features]

print(clf.predict(test_x))
