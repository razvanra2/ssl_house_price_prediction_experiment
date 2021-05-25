from DataImporter import DataImporter
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
import pandas as pd
import numpy as np


EXPERIMENT = False

def main():
    #########################
    #    Initialize data    #
    #########################
    importer = DataImporter()
    importer.ProcessData()

    x_train_scaled, y_train = importer.GetExperimentScaledTrainPair()

    #########################
    #    Experiment         #
    #########################
    if EXPERIMENT:
        models = [
            ('MLPR', MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', hidden_layer_sizes=9, learning_rate='constant', learning_rate_init=0.001, max_iter=500, solver='lbfgs')),
            ('EmptyXGBoost', xgb.XGBRegressor()),
            ('EmptyXGBRFRegressor', xgb.XGBRFRegressor),
            ('RandomForestRegressor', RandomForestRegressor()),
            ('GradientBoostingRegressor', GradientBoostingRegressor()),
            ('BayesianRidge', BayesianRidge()),
            ('HuberRegressor', HuberRegressor()),
            ('AdaBoostRegressor', AdaBoostRegressor()),
            ('ExtraTreesRegressor', ExtraTreesRegressor()),
            ('LassoLars', linear_model.LassoLars()),
            ('KernelRidge', KernelRidge())
        ]

        for modelname, model in models:
            score = cross_val_score(model, x_train_scaled, y=y_train, cv = 5, scoring="r2")
            print(modelname + " raw results (r2): ")
            print(score)
            print(modelname + " score: {:.4f} std: ({:.4f})\n".format(score.mean(), score.std()))

    #####################################
    #    Replicaate best models         #
    #####################################

    regressor = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
    x_test_scaled, ids = importer.GetExperimentScaledTestData()
    regressor.fit(x_train_scaled, y_train)
    y_predicted_reg1 = np.expm1(regressor.predict(x_test_scaled))

    regressor = RandomForestRegressor()
    x_test_scaled, ids = importer.GetExperimentScaledTestData()
    regressor.fit(x_train_scaled, y_train)
    y_predicted_reg2 = np.expm1(regressor.predict(x_test_scaled))

    regressor = BayesianRidge()
    x_test_scaled, ids = importer.GetExperimentScaledTestData()
    regressor.fit(x_train_scaled, y_train)
    y_predicted_reg3 = np.expm1(regressor.predict(x_test_scaled))

    regressor = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

    x_test_scaled, ids = importer.GetExperimentScaledTestData()
    regressor.fit(x_train_scaled, y_train)
    y_predicted_reg4 = np.expm1(regressor.predict(x_test_scaled))

    y_predicted = []
    for i in range(len(y_predicted_reg1)):
        y1 = y_predicted_reg1[i]
        y2 = y_predicted_reg2[i]
        y3 = y_predicted_reg3[i]
        y4 = y_predicted_reg4[i]
        y_predicted.append(0.70 * y1 + 0.05 * y2 + 0.00 * y3 + 0.25 * y4)

    #####################################
    #    Serialize submission           #
    #####################################

    sub = pd.DataFrame()
    sub['Id'] = ids
    sub['SalePrice'] = y_predicted
    sub.to_csv('submission.csv',index=False)

if __name__ == "__main__":
    main()
