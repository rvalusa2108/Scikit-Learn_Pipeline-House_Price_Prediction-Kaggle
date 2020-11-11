
import yaml
import os
import pprint
import pickle
from pprint import pformat
import pandas as pd
pd.options.display.max_rows=999
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
import numpy as np
import arrow
from collections import OrderedDict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, make_scorer
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
np.random.seed(42)
import Custom_Transformers_Methods as ctm

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler = logging.FileHandler('house_price_prediction_regression.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.info(f"####################### {arrow.now().format('MM/DD/YYYY HH:mm:ss - dddd, MMMM,YYYY')} ################################")

# data_train = pd.read_csv('./house-prices-advanced-regression-techniques/House_Price_Final_Dataset.csv')
# data_test  = pd.read_csv('./house-prices-advanced-regression-techniques/test.csv')
train_data = pd.read_csv('./house-prices-advanced-regression-techniques/House_Price_Final_Dataset.csv')
test_data  = pd.read_csv('./house-prices-advanced-regression-techniques/test.csv')

id_feat = ['Id']
target_feat = ['SalePrice']

X = train_data.drop(labels=target_feat+id_feat, axis=1).reset_index(drop=True)
# X = train_data.drop(labels=target_feat, axis=1).reset_index(drop=True)
y = train_data[target_feat[0]].reset_index(drop=True)

numerical_vars_list, categorical_vars_list, string_vars_list, temporal_vars_list = ctm.dataset_datatypes_counts(X)

logger.info(f"""\nTrain dataset columns with null values: {np.count_nonzero(X.isnull().sum().sort_values(ascending=False).values)}
{X.isnull().sum().sort_values(ascending=False)}""")

s = X[numerical_vars_list].isnull().sum()
logger.info(f"""
:: Numerical Features with null values ::
{s[s>0].sort_values(ascending=False)}""")

s = X[categorical_vars_list].isnull().sum()
logger.info(f"""
:: Categorical Features with null values ::
{s[s>0].sort_values(ascending=False)}""")

cat_feature_imputer_dict = OrderedDict({'Electrical': {'missing_values': np.nan
                                          ,'strategy': 'most_frequent'},
                                      'MasVnrType': {'missing_values': np.nan
                                          ,'strategy': 'most_frequent'},
                                      'GarageYrBlt': {'missing_values': np.nan
                                          ,'strategy': 'most_frequent'},
                                     })

cat_lbl_encode_list = ['MSZoning', 'Alley',
                        'LotShape', 'LandContour', 'LotConfig',
                        'Neighborhood', 'Condition1', 'BldgType',
                        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                        'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
                        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                        'Fence', 'SaleType', 'SaleCondition']

cat_1hot_encode_list = ['MSSubClass', 'MSZoning', 'Alley', 'LotShape', 'LandContour',
       'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle',
       'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence',
       'SaleType', 'SaleCondition']

feature_selection_dict = {'model_type': 'Regression',
                          'threshold': 0.0001}


analyze_feature_transform_pipeline = Pipeline([
('Cat_SimpleImputer', ctm.Custom_SimpleImputer(feature_imputer_dict=cat_feature_imputer_dict, verbose=True)),
('Cat_LabelEncoder', ctm.Custom_LabelEncoder(feature_lbl_encode_list=cat_lbl_encode_list, loginfo=True)),
('Cat_OneHotEncoder', ctm.Custom_OneHotEncoder(feature_1hot_encode_list=cat_1hot_encode_list, drop_first=False, handle_unknown='ignore', loginfo=True)),
('Feat_Selection', ctm.Custom_Feature_Selection(feature_selection_dict=feature_selection_dict, loginfo=True)),
])


breakpoint()
transformed_df = analyze_feature_transform_pipeline.fit_transform(X, y)
logger.info(f"\nFinal Tranformed Dataframe:\n{transformed_df.to_string()}")
logger.info(f"\nFinal Tranformed Dataframe shape:\n{transformed_df.shape}")





feature_transform_pipeline = Pipeline([
('Cat_SimpleImputer', ctm.Custom_SimpleImputer(feature_imputer_dict=cat_feature_imputer_dict, verbose=False)),
('Cat_LabelEncoder', ctm.Custom_LabelEncoder(feature_lbl_encode_list=cat_lbl_encode_list, loginfo=False)),
('Cat_OneHotEncoder', ctm.Custom_OneHotEncoder(feature_1hot_encode_list=cat_1hot_encode_list, drop_first=False, handle_unknown='ignore', loginfo=False)),
('Feat_Selection', ctm.Custom_Feature_Selection(feature_selection_dict=feature_selection_dict, loginfo=False)),
])


model_perf_tuning_df = ctm.model_perf_tuning(X=X,  #X=X_train,
                                y=y, #y=y_train,
                                feature_trans=feature_transform_pipeline,
                                estimator_list=['RandomForestRegressor',
                                                'GradientBoostingRegressor',
                                                #'LGBMClassifier',
                                                #'GradientBoostingClassifier',
                                                #'CatBoostClassifier',
                                                ],
                                model_type='Regression',
                                score_eval='rmsle',
                                greater_the_better=False,
                                cv_n_splits=3,
                                randomsearchcv_n_iter=1)




all_model_eval_df, best_model = ctm.model_ensemble(X=X, #X=X_train,
                                y=y, #y=y_train,
                                feature_trans=feature_transform_pipeline,
                                estimator_list=['RandomForestRegressor',
                                                'GradientBoostingRegressor',
                                                #'LGBMClassifier',
                                                #'GradientBoostingClassifier',
                                                #'RandomForestClassifier',
                                                #'CatBoostClassifier'
                                                 ],
                                model_type='Regression',
                                score_eval='rmsle',
                                greater_the_better=False,
                                model_perf_tuning_df=model_perf_tuning_df)

test_dataset = test_data.drop(labels=id_feat, axis=1).reset_index(drop=True)
transformed_test_dataset = analyze_feature_transform_pipeline.transform(test_dataset)
logger.info(f"""\nTransformed Test Dataset:\n{transformed_test_dataset.to_string()} """)
logger.info(f"""\nTransformed Test Dataset shape:\n{transformed_test_dataset.shape} """)

best_model = 'rfr'
# Training the best model with the complete dataset to pickle the final best model for test data prediction
ctm.final_model_training(complete_X_train=X,
                        complete_y_train=y,
                        best_model=best_model)

# Final Test Data Predictions
with open('final_best_model.plk', 'rb') as f:
    best_model_pipeline = pickle.load(f)

test_pred = best_model_pipeline.predict(test_data)


#============================================================================
#Kaggle Submission:
#Check this for more info-https://technowhisp.com/kaggle-api-python-documentation/
if True:
#if False:
    test_df = test_data
    test_pred = test_pred
    id_feature = id_feat[0]
    target_feature = target_feat[0]
    sub_file_name = 'house_price_prediction'
    submission_msg = 'Final_Test_Data_Submission'
    competition_name = 'house-prices-advanced-regression-techniques'
    ctm.kaggle_submission(test_df=test_df,
                      test_pred=test_pred,
                      id_feature=id_feature,
                      target_feature=target_feature,
                      sub_file_name=sub_file_name,
                      submission_msg=submission_msg,
                      competition_name=competition_name)










#========================================================


from sklearn.preprocessing import LabelEncoder
import numpy as np


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)


country_list = ['Argentina', 'Australia', 'Canada', 'France',
                'Italy', 'Spain', 'US', 'Canada', 'Argentina, ''US']

label_encoder = LabelEncoderExt()

label_encoder.fit(country_list)
print(label_encoder.classes_) # you can see new class called Unknown
print(label_encoder.transform(country_list))


new_country_list = ['Canada', 'France', 'Italy', 'Spain', 'US', 'India',
                    'Pakistan', 'South Africa']
print(label_encoder.transform(new_country_list))





















