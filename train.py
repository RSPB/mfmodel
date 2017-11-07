import time
import configobj
import xgboost as xg
from validate import Validator
from sklearn.model_selection import train_test_split


def split_data(X, y, val_fraction, test_fraction, seed=42):
    if isinstance(y, str):
        y = X.pop(y)
    if test_fraction > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=seed, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_fraction, random_state=seed, stratify=y_train)
        dtest = xg.DMatrix(X_test, label=y_test)
        dtrain = xg.DMatrix(X_train, label=y_train)
        dval = xg.DMatrix(X_val, label=y_val)
        return dtrain, dval, dtest
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_fraction, random_state=seed, stratify=y)
        dtrain = xg.DMatrix(X_train, label=y_train)
        dval = xg.DMatrix(X_val, label=y_val)
        return dtrain, dval


def train_model(dtrain, dval, params=None, boost_rounds=500, early_stopping_rounds=5):
    if not params:
        config = configobj.ConfigObj('xgboost_params.ini', configspec='xgboost_params_specs.ini')
        assert config.validate(Validator()), 'xgboost_params.ini contains error'
        params = config['xgboost']

    evallist = [(dval, 'eval'), (dtrain, 'train')]
    model = xg.train(params=params,
                     dtrain=dtrain,
                     num_boost_round=boost_rounds,
                     evals=evallist,
                     early_stopping_rounds=early_stopping_rounds)
    return model


def main():
    t0 = time.time()

    config = configobj.ConfigObj('xgboost_params.ini', configspec='xgboost_params_specs.ini')
    validation_successful = config.validate(Validator())
    print(config['xgboost'])

    print('Run time: {:.2f} s'.format(time.time() - t0))

if __name__ == '__main__':
    main()