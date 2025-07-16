#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





df=pd.read_csv("creditcard.csv")





df.head()





df.info()





df.describe()





df.isna().sum()





df.nunique()





df.shape





df.drop_duplicates()





df.shape





plt.figure(figsize=(28,32))
for i in range (1, 29):
    plt.subplot(7, 4, i)
    plt.hist(df[f'V{i}'], bins=50, edgecolor='black')
    plt.title(f'V{i} Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')





plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, cbar_kws={'shrink':0.8})
plt.title('Correlation Heatmap of V1 to V28')
plt.show()





fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(df[df['Class'] == 1])))
print('Valid Transactions: {}'.format(len(df[df['Class'] == 0])))





print("Amount details of the fraudulent transaction")
fraud.Amount.describe()





print("details of valid transaction")
valid.Amount.describe()


# In[ ]:


X = df.drop(['Class'], axis = 1)
Y = df["Class"]
print(X.shape)
print(Y.shape)

xData = X.values
yData = Y.values

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_rfc = RandomForestClassifier(criterion='gini', random_state=42)





model_rfc.fit(xTrain, yTrain)





yPred = model_rfc.predict(xTest)





accu=model_rfc.score(xTest, yTest) * 100
accu


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix 
accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()