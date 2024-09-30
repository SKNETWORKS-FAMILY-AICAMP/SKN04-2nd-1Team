from src.data import CostomerDataset, CostomerDataModule
from src.utils import convert_category_into_integer
from src.model.mlp import Model
from src.training import CostomerModule

import pandas as pd
import numpy as np
import random
import json
import nni

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import torch

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger



def main(configs):
    data = pd.read_csv('./data/train.csv')
    data = data.drop(columns=[
        'CustomerID',  # 구분용도임
        'NotNewCellphoneUser', # NewCell...이랑 중복
        'BlockedCalls', 
        'PrizmCode', # 인구통계 세분화 코드라 삭제
        'TruckOwner', # 자동차 오토바이 유무라서 삭제
        'OwnsMotorcycle',
        'OwnsComputer', # 컴퓨터 유무 삭제
        'OffPeakCallsInOut',
        'OptOutMailings',
        'NonUSTravel',# 미국 여행여부 삭제
        'AdjustmentsToCreditRating', # 신용등급
        'ActiveSubs',
    ])
    data = data.dropna()

    # 나이 0 삭제
    data = data[data['AgeHH1']>18]
    data = data[data['AgeHH2']>18]

    # 인트타입으로 바꿔서 구분
    data['Churn'] = np.where(data['Churn']=="Yes", 1, 0)
    data['DroppedCalls'] = data['DroppedCalls'].astype(int)

    # 나이 구분
    data.AgeHH1 = np.where(
        data.AgeHH1 < 20,18,
        np.where(data.AgeHH1 < 30, 20, 
        np.where(data.AgeHH1 < 40, 30,
        np.where(data.AgeHH1 < 50, 40,
        np.where(data.AgeHH1 < 60, 50, 60)))))
    data.AgeHH2 = np.where(
        data.AgeHH2 < 20,18,
        np.where(data.AgeHH2 < 30, 20, 
        np.where(data.AgeHH2 < 40, 30,
        np.where(data.AgeHH2 < 50, 40,
        np.where(data.AgeHH2 < 60, 50, 60)))))

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
        data['CreditRating'] > data['CreditRating'].median(), 
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
    category_columns = ['ServiceArea','ChildrenInHH','HandsetRefurbished','HandsetWebCapable',
                        'RVOwner','Homeownership','BuysViaMailOrder','RespondsToMailOffers',
                        'HasCreditCard','NewCellphoneUser','HandsetPrice','MadeCallToRetentionTeam',
                        'Occupation','MaritalStatus','CustomerLoyalty','EquipmentUsageDuration','ChargeBurden',
                        'CreditCategory','MarketingEngagement','HandsetWebCapability']
    
    data, _ = convert_category_into_integer(data, (category_columns))
    data = data.astype(np.float32)

    # 학습, 검증, 테스트 데이터셋 분할
    train, temp = train_test_split(data, test_size=0.4, random_state=seed)
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    # Dataset 생성
    train_dataset = CostomerDataset(train)
    valid_dataset = CostomerDataset(valid)
    test_dataset = CostomerDataset(test)

    costomer_data_module = CostomerDataModule(batch_size=configs.get('batch_size'))
    costomer_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    robust_scaler = RobustScaler()
    
    need_scale_cols = data.columns.difference(category_columns + ['RetentionCalls', 'RetentionOffersAccepted', 'IncomeGroup'])
    
    train.loc[:, need_scale_cols] = \
        robust_scaler.fit_transform(train.loc[:, need_scale_cols])

    # 검증 데이터와 테스트 데이터의 열을 훈련 데이터의 통계로 표준화
    valid.loc[:, need_scale_cols] = \
        robust_scaler.transform(valid.loc[:, need_scale_cols])

    test.loc[:, need_scale_cols] = \
        robust_scaler.transform(test.loc[:, need_scale_cols])

    configs.update({'input_dim': len(data.columns)-1})
    model = Model(configs)

    costomer_module = CostomerModule(
        model=model,
        configs=configs,
    )

    # del configs['output_dim'], configs['seed']
    exp_name = 'costomer'
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=5),
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'costomer/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    trainer.fit(
        model=costomer_module,
        datamodule=costomer_data_module,
    )
    
    trainer.test(
        model=costomer_module,
        datamodule=costomer_data_module,
    )
    
    torch.save(model.state_dict(), './model/mlp.pth')
    
if __name__ == '__main__':
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    if configs.get('nni'):
        nni_params = nni.get_next_parameter()
        configs.update({'batch_size': nni_params.get('batch_size')})
        configs.update({'hidden_dim': nni_params.get('hidden_dim')})
        configs.update({'learning_rate': nni_params.get('learning_rate')})
        configs.update({'dropout_ratio': nni_params.get('dropout_ratio')})

    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    main(configs)