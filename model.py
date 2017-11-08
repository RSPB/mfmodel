import time
import configobj
import logging
import xgboost as xg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
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


def load_params(config_path, config_specs_path):
    try:
        config = configobj.ConfigObj(config_path, configspec=config_specs_path)
    except:
        logging.exception('Unable to load xgboost parameters from %s (type specs in %s)', config_path, config_specs_path)
        raise
    assert config.validate(Validator()), '{} contains errors'.format(config_path)
    params = config['xgboost']
    return params


def train(dtrain, dval, params=None, boost_rounds=500, early_stopping_rounds=5, saveto='model.xgb'):
    if not params:
        params = load_params(config_path='xgboost_params.ini', config_specs_path='xgboost_params_specs.ini')

    evallist = [(dval, 'eval'), (dtrain, 'train')]
    model = xg.train(params=params,
                     dtrain=dtrain,
                     num_boost_round=boost_rounds,
                     evals=evallist,
                     early_stopping_rounds=early_stopping_rounds)
    if saveto:
        model.save_model(saveto)
    return model

def predict(features, model_path):
    booster = xg.Booster()
    booster.load_model(model_path)
    prediction = booster.predict(features)


def _remove_spaces_in_feature_names(X):
    return [name.replace(' ', '') for name in X.feature_names]


def evaluate(model, dtest, figure_name=None):
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_test = dtest.get_label().astype(int)

    precision_avg, recall_avg, _, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='micro')
    cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

    model.feature_names = _remove_spaces_in_feature_names(model)

    results = {
        'accuracy': metrics.accuracy_score(y_true=y_test, y_pred=y_pred),
        'classification_report': metrics.classification_report(y_true=y_test, y_pred=y_pred),
        'confusion_matrix': cm,
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
    }

    if figure_name:
        precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_prob)
        f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 30), gridspec_kw={'height_ratios': [1, 2, 3]})
        ax1.step(recall, precision, color='b', alpha=0.2, where='post')
        ax1.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlim([0.0, 1.0])
        ax1.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(precision_avg))
        sns.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g', square=True,
                    xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'], ax=ax2)
        ax2.set_title('Confusion matrix')
        xg.plot_importance(model, ax3)
        f.savefig(figure_name)

    return results


def main():
    t0 = time.time()

    config = configobj.ConfigObj('xgboost_params.ini', configspec='xgboost_params_specs.ini')
    validation_successful = config.validate(Validator())
    print(config['xgboost'])

    print('Run time: {:.2f} s'.format(time.time() - t0))


if __name__ == '__main__':
    main()