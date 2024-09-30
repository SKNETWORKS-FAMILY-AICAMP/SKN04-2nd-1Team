import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

seed= 42

data = pd.read_csv('./data/train.csv', usecols=['Churn','TotalRecurringCharge', 'MonthsInService', 'CurrentEquipmentDays'])
data = data.dropna()

# 'CurrentEquipmentDays'를 사용해 서비스를 얼마나 사용하고 있는지 반영
data['EquipmentUsageDuration'] = np.where(
    data['CurrentEquipmentDays'] > data['CurrentEquipmentDays'].mean(), 
    'Long-Term Equipment User', 'Short-Term Equipment User'
)

label_encoder = LabelEncoder()

# 레이블 인코딩을 적용할 열 목록
columns_to_encode = [
'EquipmentUsageDuration',
]

# 각 열에 대해 레이블 인코딩 수행
for col in columns_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Churn == No 에대한 recall, precision
results_0 = {}
stratified_k_fold = StratifiedKFold(random_state=seed, shuffle=True)
for i, (train_index, test_index) in enumerate(stratified_k_fold.split(data, data.Churn)):
    temp = data.iloc[train_index]
    test = data.iloc[test_index]
    train, valid = train_test_split(temp, test_size=len(test)/len(temp))

    lgbm = LGBMClassifier(verbose=0,random_state=seed)
    lgbm.fit(
        train.drop(columns=['Churn']), train.Churn,
        eval_set=(valid.drop(columns=['Churn']), valid.Churn)
    )

    confusion_matrix_result = confusion_matrix(
        test.Churn,
        lgbm.predict(test.drop(columns=['Churn'])),
    )
    TP = confusion_matrix_result[0, 0]
    FN = confusion_matrix_result[0, 1]
    FP = confusion_matrix_result[1, 0]
    TN = confusion_matrix_result[1, 1]

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1 = 2*precision*recall / (precision+recall)
    accuracy = (TP+TN) / (TP+TN+FP+FN)

    results_0.update({
        f'fold{i+1}': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
        }
    })

# Churn == Yes 에대한 recall, precision
results_1 = {}
stratified_k_fold = StratifiedKFold(random_state=seed, shuffle=True)
for i, (train_index, test_index) in enumerate(stratified_k_fold.split(data, data.Churn)):
    temp = data.iloc[train_index]
    test = data.iloc[test_index]
    train, valid = train_test_split(temp, test_size=len(test)/len(temp))

    lgbm = LGBMClassifier(verbose=0,random_state=seed)
    lgbm.fit(
        train.drop(columns=['Churn']), train.Churn,
        eval_set=(valid.drop(columns=['Churn']), valid.Churn)
    )

    confusion_matrix_result = confusion_matrix(
        test.Churn,
        lgbm.predict(test.drop(columns=['Churn'])),
        labels=[1,0]
    )
    TP = confusion_matrix_result[0, 0]
    FN = confusion_matrix_result[0, 1]
    FP = confusion_matrix_result[1, 0]
    TN = confusion_matrix_result[1, 1]

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1 = 2*precision*recall / (precision+recall)
    accuracy = (TP+TN) / (TP+TN+FP+FN)

    results_1.update({
        f'fold{i+1}': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
        }
    })

print('Churn == No 에대한 recall, precision')
print('-'*30)
print(results_0)
print('-'*30)

print('Churn == Yes 에대한 recall, precision')
print('-'*30)
print(results_1)