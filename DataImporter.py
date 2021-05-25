import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
from sklearn import preprocessing
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from scipy import stats
from scipy.stats import norm, skew #for some statistics
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
from subprocess import check_output
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
class DataImporter():
    def __init__(self):
        # read input from dataset
        self.train = pd.read_csv('train.csv')

        #Deleting outliers
        self.train = self.train.drop(self.train[(self.train['GrLivArea']>4000) & (self.train['SalePrice']<300000)].index)
        #Log-transform target variable to form norm distribution
        self.train["SalePrice"] = np.log1p(self.train["SalePrice"])

        self.test = pd.read_csv('test.csv')
        self.ntrain = self.train.shape[0]
        self.ntest = self.test.shape[0]

    def ProcessData(self):
        self.train_ID = self.train['Id']
        self.test_ID = self.test['Id']

        self.train.drop("Id", axis = 1, inplace = True)
        self.test.drop("Id", axis = 1, inplace = True)

        self.y_train = self.train.SalePrice.values

        self.all_data = pd.concat((self.train, self.test)).reset_index(drop=True)


        #########################################
        # Feature Engineering and Vrajeala Fina #
        #########################################
        # usually houses don't have pools so it's safe to assume N/A means None
        self.all_data["PoolQC"] = self.all_data["PoolQC"].fillna("None")

        # same as pools
        self.all_data["MiscFeature"] = self.all_data["MiscFeature"].fillna("None")

        # N/A means no alley access
        self.all_data["Alley"] = self.all_data["Alley"].fillna("None")

        # as per data description
        self.all_data["Fence"] = self.all_data["Fence"].fillna("None")

        # No fireplace
        self.all_data["FireplaceQu"] = self.all_data["FireplaceQu"].fillna("None")

        #Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
        self.all_data["LotFrontage"] = self.all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

        # Garage features changes
        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            self.all_data[col] = self.all_data[col].fillna('None')
        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
            self.all_data[col] = self.all_data[col].fillna(0)

        # Basements
        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            self.all_data[col] = self.all_data[col].fillna(0)
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            self.all_data[col] = self.all_data[col].fillna('None')

        # I don't even know what this is lol, must be boujee stuff
        self.all_data["MasVnrType"] = self.all_data["MasVnrType"].fillna("None")
        self.all_data["MasVnrArea"] = self.all_data["MasVnrArea"].fillna(0)

        # Zoning data
        self.all_data['MSZoning'] = self.all_data['MSZoning'].fillna(self.all_data['MSZoning'].mode()[0])

        self.all_data = self.all_data.drop(['Utilities'], axis=1)

        self.all_data["Functional"] = self.all_data["Functional"].fillna("Typ")

        self.all_data['Electrical'] = self.all_data['Electrical'].fillna(self.all_data['Electrical'].mode()[0])

        self.all_data['KitchenQual'] = self.all_data['KitchenQual'].fillna(self.all_data['KitchenQual'].mode()[0])

        self.all_data['Exterior1st'] = self.all_data['Exterior1st'].fillna(self.all_data['Exterior1st'].mode()[0])
        self.all_data['Exterior2nd'] = self.all_data['Exterior2nd'].fillna(self.all_data['Exterior2nd'].mode()[0])

        self.all_data['SaleType'] = self.all_data['SaleType'].fillna(self.all_data['SaleType'].mode()[0])

        self.all_data['MSSubClass'] = self.all_data['MSSubClass'].fillna("None")

        # Adding total sqfootage feature
        self.all_data['TotalSF'] = self.all_data['TotalBsmtSF'] + self.all_data['1stFlrSF'] + self.all_data['2ndFlrSF']

        self.all_data = self.all_data.drop(['ScreenPorch'], axis=1)
        self.all_data = self.all_data.drop(['ExterCond'], axis=1)
        self.all_data = self.all_data.drop(['BsmtFinSF2'], axis=1)
        self.all_data = self.all_data.drop(['BsmtHalfBath'], axis=1)
        self.all_data = self.all_data.drop(['SaleType'], axis=1)
        self.all_data = self.all_data.drop(['Heating'], axis=1)
        self.all_data = self.all_data.drop(['LotFrontage'], axis=1)
        self.all_data = self.all_data.drop(['RoofMatl'], axis=1)
        self.all_data = self.all_data.drop(['MiscVal'], axis=1)
        self.all_data = self.all_data.drop(['YrSold'], axis=1)
        self.all_data = self.all_data.drop(['LotConfig'], axis=1)
        self.all_data = self.all_data.drop(['Condition2'], axis=1)
        self.all_data = self.all_data.drop(['MoSold'], axis=1)
        self.all_data = self.all_data.drop(['PoolArea'], axis=1)
        self.all_data = self.all_data.drop(['PoolQC'], axis=1)



        #########################################
        # End region                            #
        #########################################


        self.all_data.drop(['SalePrice'], axis=1, inplace=True)

        objcolslist = []
        for col in self.all_data:
            if (self.all_data[col].dtype == "object"):
                objcolslist.append(col)

        '''
        self.all_data = pd.get_dummies(self.all_data, columns = objcolslist)
        print(self.all_data)
        print(self.all_data.describe())
        '''

        labelEncoder = LabelEncoder()
        for col in objcolslist:
            self.all_data[col] = labelEncoder.fit_transform(self.all_data[col])

        imputer = SimpleImputer(missing_values= np.nan, strategy='median')

        idf = pd.DataFrame(imputer.fit_transform(self.all_data))
        idf.columns = self.all_data.columns
        idf.index = self.all_data.index
        self.all_data = idf

        min_max_scaler = preprocessing.StandardScaler()
        self.x_train = self.all_data.values[:self.ntrain]
        self.x_train_scaled = min_max_scaler.fit_transform(self.all_data.values)[:self.ntrain]

        min_max_scaler = preprocessing.StandardScaler()
        self.x_test = self.all_data.values[self.ntrain:]
        self.x_test_scaled = min_max_scaler.fit_transform(self.all_data.values)[self.ntrain:]


    def GetExperimentScaledTrainPair(self):
        return self.x_train_scaled, self.y_train

    def GetExperimentScaledTestData(self):
        return self.x_test_scaled, self.test_ID

