import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('./data/train.csv', usecols=['Churn','TotalRecurringCharge', 'MonthsInService', 'CurrentEquipmentDays','CreditRating','RetentionCalls','RespondsToMailOffers','BuysViaMailOrder','HandsetWebCapable'])
data = data.dropna()
data.CreditRating = data.CreditRating.str[0].astype(int) # 신용등급 숫자로 변경

# 1. 고객 충성도 레이블 (Customer Loyalty)
# 'MonthsInService'와 'RetentionCalls'을 기반으로 고객이 얼마나 오랜 기간 동안 서비스에 머물렀는지, 그리고 고객 유지 노력의 결과를 반영
data['CustomerLoyalty'] = np.where(
    (data['MonthsInService'] > data['MonthsInService'].mean()) &
    (data['RetentionCalls'] > 0), 
    'High Loyalty', 'Low Loyalty'
)

# 2. 서비스 사용 기간 레이블 (Equipment Usage Duration)
# 'CurrentEquipmentDays'를 사용해 서비스를 얼마나 사용하고 있는지 반영
data['EquipmentUsageDuration'] = np.where(
    data['CurrentEquipmentDays'] > data['CurrentEquipmentDays'].mean(), 
    'Long-Term Equipment User', 'Short-Term Equipment User'
)

# 3. 요금 부담 레이블 (Charge Burden)
# 'TotalRecurringCharge'를 기준으로 요금 부담이 높은지 낮은지를 분류
data['ChargeBurden'] = np.where(
    data['TotalRecurringCharge'] > data['TotalRecurringCharge'].mean(), 
    'High Charge', 'Low Charge'
)

# 4. 신용 등급 레이블 (Credit Rating Category)
# 'CreditRating'을 기준으로 신용 등급을 두 그룹으로 나눔
data['CreditCategory'] = np.where(
    data['CreditRating'] < 6,
    'High Credit', 'Low Credit'
)

# 5. 구매 및 메일 응답 행동 (Purchase and Mail Response Behavior)
# 'BuysViaMailOrder'와 'RespondsToMailOffers'를 결합하여 고객의 마케팅 참여도 파악
data['MarketingEngagement'] = np.where(
    (data['BuysViaMailOrder'] == 'Yes') & (data['RespondsToMailOffers'] == 'Yes'), 
    'Fully Engaged',
    np.where(
        (data['BuysViaMailOrder'] == 'Yes') | (data['RespondsToMailOffers'] == 'Yes'), 
        'Partially Engaged', 'Not Engaged'
    )
)

# 6. 핸드셋 웹 사용 가능 여부 (Handset Web Capability)
# 'HandsetWebCapable'을 사용하여 핸드셋이 웹 사용 가능한지 여부를 분류
data['HandsetWebCapability'] = np.where(
    data['HandsetWebCapable'] == 'Yes', 
    'WebCapable', 'NonWebCapable'
)

label_encoder = LabelEncoder()

# 레이블 인코딩을 적용할 열 목록
columns_to_encode = [
'HandsetWebCapable',
'BuysViaMailOrder',
'RespondsToMailOffers',
'CustomerLoyalty',
'EquipmentUsageDuration',
'ChargeBurden',
'CreditCategory',
'MarketingEngagement',
'HandsetWebCapability'
]

# 각 열에 대해 레이블 인코딩 수행
for col in columns_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# 피처와 레이블 분리
X = data.drop(columns=['Churn'])
y = data['Churn']

# 데이터 나누기 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM 모델 설정 (하이퍼파라미터 조정)
model = LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    boosting_type='gbdt',
    num_leaves=40,  # num_leaves 수치를 줄여 모델 복잡성을 낮춤
    learning_rate=0.03,  # 학습률을 약간 높임
    feature_fraction=0.7,  # 사용되는 특성 비율을 높여서 학습 속도를 개선
    bagging_fraction=0.7,  # 샘플링 비율을 높여서 학습 속도를 개선
    bagging_freq=2,  # 가방 샘플링의 빈도를 높임
    n_estimators=700,  # 부스팅 라운드 수를 줄임
    early_stopping_rounds=50,  # 조기 종료 수치를 낮춤
    min_data_in_leaf=15,  # 리프 노드당 최소 데이터 수를 낮춤
    max_depth=10,  # 최대 깊이를 제한하여 과적합 방지
    lambda_l1=0.0,
    lambda_l2=0.5,  # L2 정규화를 약간 줄임
)

# 모델 학습
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss', 
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)])

# 예측
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # 확률 예측

# 혼동 행렬 및 분류 리포트 출력
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))