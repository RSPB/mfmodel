import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
from sklearn import metrics

def _remove_spaces_in_feature_names(X):
    return [name.replace(' ', '') for name in X.feature_names]


def evaluate_model(model, dtest, figure_name=None):
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
        plot_importance(model, ax3)
        f.savefig(figure_name)

    return results