import time
import logging
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split
from appconfig import setup_logging
from sklearn import metrics

params = {'max_depth': 13,
          'n_estimators': 1000,
          'objective': 'binary:logistic',
          'eval_metric': ['auc', 'error'],
          'nthread': 15}


def main():
    t0 = time.time()
    num_parallel = 14
    equal_no_samples_in_each_class = False
    datapath = '/home/tracek/Data/gender/gender_warbler.csv'
    datapath = '/home/tracek/Data/gender/gender_descriptors.csv'

    setup_logging()
    data = pd.read_csv(datapath).drop(['filename'], axis=1)  # centroid corresponds to meanfreq
    male_df_len = len(data[data['label'] == 0])
    female_df_len = len(data[data['label'] == 1])

    if equal_no_samples_in_each_class:
        fraction_to_drop = 1 - female_df_len / male_df_len
        data = data.drop(data[data['label'] == 0].sample(frac=fraction_to_drop, random_state=42).index)

    y = data.pop('label')
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1)

    dtrain = xg.DMatrix(X_train, label=y_train)
    dtest = xg.DMatrix(X_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    model = xg.train(params=params, dtrain=dtrain, num_boost_round=30, evals=evallist)
    # y_pred = clf.predict(X_test)
    # r = metrics.classification_report(y_true=y_test, y_pred=y_pred)
    # print(r)
    # accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # print('Accuracy: {:.3f}%'.format(accuracy * 100))

    print('Run time: {:.2f} s'.format(time.time() - t0))

if __name__ == '__main__':
    main()