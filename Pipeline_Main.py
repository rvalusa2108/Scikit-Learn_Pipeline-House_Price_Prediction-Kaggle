
import yaml
import os
import pprint
from pprint import pformat
import pandas as pd
pd.options.display.max_rows=999
pd.set_option('display.max_columns', 500)
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

data_train = pd.read_csv('./house-prices-advanced-regression-techniques/House_Price_Final_Dataset.csv')
data_test  = pd.read_csv('./house-prices-advanced-regression-techniques/test.csv')

#id_feat = ['Id']
target_feat = ['SalePrice']

#X = data_train.drop(labels=target_feat+id_feat, axis=1).reset_index(drop=True)
X = data_train.drop(labels=target_feat, axis=1).reset_index(drop=True)
y = data_train[target_feat[0]].reset_index(drop=True)

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

# outlier_detection_transform_dict = {'features_list': numerical_vars_list,
#                                     'min_outliers': 2,
#                                     'drop_outliers': True}

# X, y = ctm.Outlier_Detection(X, y, outlier_detection_transform_dict)

# num_feature_imputer_dict = OrderedDict({'MasVnrArea': {'missing_values': np.nan
#                                           ,'strategy': 'median'},
#                                       'GarageYrBlt': {'missing_values': np.nan
#                                           ,'strategy': 'median'},
#                                       'LotFrontage': {'missing_values': np.nan
#                                           ,'strategy': 'median'}
#                                      })

# cat_feature_imputer_dict = OrderedDict({'FireplaceQu': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'NotAvailable'},
#                                       'MasVnrType': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'None'},
#                                       'BsmtQual': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'No_Basement'},
#                                       'BsmtCond': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'No_Basement'},
#                                       'BsmtExposure': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'No_Basement'},
#                                       'BsmtFinType1': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'No_Basement'},
#                                       'BsmtFinType2': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'No_Basement'},
#                                       'Electrical': {'missing_values': np.nan
#                                           ,'strategy': 'most_frequent'},
#                                       'GarageType': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'No_Garage'},
#                                       'GarageFinish': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'No_Garage'},
#                                       'GarageQual': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'No_Garage'},
#                                       'GarageCond': {'missing_values': np.nan
#                                           ,'strategy': 'constant',
#                                           'constant_value': 'No_Garage'},
#                                      })


cat_feature_imputer_dict = OrderedDict({'Electrical': {'missing_values': np.nan
                                          ,'strategy': 'most_frequent'},
                                      'MasVnrType': {'missing_values': np.nan
                                          ,'strategy': 'most_frequent'},
                                     })

cat_lbl_encode_list = ['MSSubClass', 'MSZoning', 'Alley',
                        'LotShape', 'LandContour', 'LotConfig',
                        'Neighborhood', 'Condition1', 'BldgType',
                        'HouseStyle', 'OverallQual', 'OverallCond',
                        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                        'MasVnrType', 'ExterQual', 'Foundation',
                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                        'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
                        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                        'Fence', 'SaleType', 'SaleCondition']

# analyze_feature_transform_pipeline = Pipeline([
# ('Drop_Feat_w_High_Missing_Values', ctm.Custom_Missing_Values_Check_Column_Drop(missing_val_percentage=0.7, loginfo=True)),
# ('Num_SimpleImputer', ctm.Custom_SimpleImputer(feature_imputer_dict=num_feature_imputer_dict, verbose=True)),
# ('Cat_SimpleImputer', ctm.Custom_SimpleImputer(feature_imputer_dict=cat_feature_imputer_dict, verbose=True)),
# ('Cat_LabelEncoder', ctm.Custom_LabelEncoder(feature_lbl_encode_list=cat_lbl_encode_list, loginfo=True)),
# ])

analyze_feature_transform_pipeline = Pipeline([
('Cat_SimpleImputer', ctm.Custom_SimpleImputer(feature_imputer_dict=cat_feature_imputer_dict, verbose=True)),
('Cat_LabelEncoder', ctm.Custom_LabelEncoder(feature_lbl_encode_list=cat_lbl_encode_list, loginfo=True)),
])


breakpoint()
transformed_df = analyze_feature_transform_pipeline.fit_transform(X)
logger.info(f"\nFinal Tranformed Dataframe:\n{transformed_df.to_string()}")
logger.info(f"\nFinal Tranformed Dataframe shape:\n{transformed_df.shape}")














feature_transform_pipeline = Pipeline([
# ('Cabin_SimpleImputer', ctm.Custom_SimpleImputer(feature_imputer_dict=cabin_imputer_dict, verbose=False)),
# ('Custom_Text_Features_Transformer', ctm.Custom_Text_Features_Transformer(text_feature_transform_dict=text_feature_transform_dict, loginfo=False)),
# ('Custom_SimpleImputer', ctm.Custom_SimpleImputer(feature_imputer_dict=feature_imputer_dict, verbose=False)),
# ('Custom_New_Features_Creation', ctm.Custom_New_Features_Creation(new_features_creation_dict)),
# ('Custom_Features_Binning_Transformer', ctm.Custom_Features_Binning_Transformer(feature_binning_dict, loginfo=False)),
# ('Custom_LabelEncoder', ctm.Custom_LabelEncoder(feature_lbl_encode_list=feature_lbl_encode_list)),
# ('Custom_OneHotEncoder', ctm.Custom_OneHotEncoder(feature_1hot_encode_list=feature_1hot_encode_list)),
])


model_perf_tuning_df = ctm.model_perf_tuning(X=X,  #X=X_train,
                                y=y, #y=y_train,
                                feature_trans=feature_transform_pipeline,
                                estimator_list=['LGBMClassifier',
                                                'GradientBoostingClassifier',
                                                'RandomForestClassifier',
                                                #'CatBoostClassifier'
                                                ],
                                score_eval=roc_auc_score)
                                #score_eval=accuracy_score)



all_tree_model_eval_df, best_tree_model = ctm.model_ensemble_classification(X=X, #X=X_train,
                                            y=y, #y=y_train,
                                            feature_trans=feature_transform_pipeline,
                                            estimator_list=['LGBMClassifier',
                                                            'GradientBoostingClassifier',
                                                            'RandomForestClassifier',
                                                            #'CatBoostClassifier'
                                                             ],
                                            score_eval=roc_auc_score,
                                            #score_eval=accuracy_score,
                                            model_perf_tuning_df=model_perf_tuning_df)


test_dataset = data_test.drop(labels=id_feat, axis=1).reset_index(drop=True)
transformed_test_dataset = analyze_feature_transform_pipeline.transform(test_dataset)
logger.info(f"""\nTransformed Test Dataset:\n{transformed_test_dataset.to_string()} """)
logger.info(f"""\nTransformed Test Dataset shape:\n{transformed_test_dataset.shape} """)
test_pred = ctm.final_tree_model_training_pred(complete_X_train=X,
                                               complete_y_train=y,
                                               test_data=test_dataset,
                                               best_tree_model=best_tree_model)


#============================================================================
#Kaggle Submission:
#Check this for more info-https://technowhisp.com/kaggle-api-python-documentation/
if True:
#if False:
    test_df = data_test
    test_pred = test_pred
    id_feature = id_feat[0]
    target_feature = target_feat[0]
    sub_file_name = 'titanic_competition'
    submission_msg = 'Final_Tree_Model'
    competition_name = 'titanic'
    ctm.kaggle_submission(test_df=test_df,
                      test_pred=test_pred,
                      id_feature=id_feature,
                      target_feature=target_feature,
                      sub_file_name=sub_file_name,
                      submission_msg=submission_msg,
                      competition_name=competition_name)




