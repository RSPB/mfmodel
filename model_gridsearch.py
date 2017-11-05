import time
import logging
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split, GridSearchCV
from appconfig import setup_logging
from sklearn import metrics


param_grid = {'max_depth': [13, 20], 'n_estimators': [200, 500, 1000], 'gamma': [0, 0.1]}


def main():
    t0 = time.time()
    n_folds = 5
    num_parallel = 14
    equal_no_samples_in_each_class = False
    datapath = '/home/tracek/Data/gender/gender_warbler.csv'

    setup_logging()
    data = pd.read_csv(datapath).drop(['centroid', 'filename'], axis=1)  # centroid corresponds to meanfreq
    male_df_len = len(data[data['label'] == 0])
    female_df_len = len(data[data['label'] == 1])
    logging.info('Male samples %d', male_df_len)
    logging.info('Female samples $d', female_df_len)

    if equal_no_samples_in_each_class:
        fraction_to_drop = 1 - female_df_len / male_df_len
        data = data.drop(data[data['label'] == 0].sample(frac=fraction_to_drop, random_state=42).index)

    y = data.pop('label')

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1)

    clf = xg.XGBClassifier(n_jobs=num_parallel, objective='binary:logistic', learning_rate=0.1)
    grid_cv = GridSearchCV(estimator=clf, param_grid=param_grid, cv=n_folds)
    grid_cv.fit(X=X_train, y=y_train)
    print(grid_cv.best_params_)

    # y_pred = clf.predict(X_test)
    # r = metrics.classification_report(y_true=y_test, y_pred=y_pred)
    # print(r)
    # accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # print('Accuracy: {:.3f}%'.format(accuracy * 100))

    print('Run time: {:.2f} s'.format(time.time() - t0))

if __name__ == '__main__':
    main()