import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, auc, accuracy_score, roc_auc_score, precision_recall_curve, precision_score, average_precision_score, \
    confusion_matrix
from statistics import mean
import time
from imblearn.over_sampling import SVMSMOTE

sm = SVMSMOTE(random_state=42, k_neighbors=2)


def perform_nested_cv(clf, X, Y, dataset, classes, params_grid=[]):
    outer_kf = StratifiedKFold(10)
    inner_kf = StratifiedKFold(3)
    best_estimator = None
    gs_cls = RandomizedSearchCV(clf,
                                refit='accuracy',
                                param_distributions=params_grid,
                                scoring=['f1_macro', 'accuracy'],
                                cv=inner_kf,
                            )
    acc_scores, roc_scores, tprs, fprs, precisions, auprs, best, train_times, infer_times = [], [], [], [], [], [], [], [], []
    counter_cv = 1
    X, Y = sm.fit_resample(X, Y)
    for train, test in outer_kf.split(X, Y): #start 10 cross validation
        X_train, X_test = X.iloc[train], X.iloc[test]
        Y_train, Y_test = Y.iloc[train], Y.iloc[test]
        gs_cls.fit(X_train, Y_train) #fit with 3 KFold hyper parameter tuning
        best_params = gs_cls.best_params_
        best.append(best_params)
        best_estimator = gs_cls.best_estimator_
        print('Best params found for CV Number: ', counter_cv, " is ", best_params)
        counter_cv += 1
        train_time = time.time()
        if not params_grid:
            best_estimator = clf.fit(X_train, Y_train)
        else:
            best_estimator.fit(X_train, Y_train)
        end_train_time = time.time() - train_time
        train_times.append(end_train_time)
        print("Validation.....")
        infer_time = time.time() # Calculations for metrics
        Y_pred = best_estimator.predict(X_test)
        proba = best_estimator.predict_proba(X_test)
        end_infer_time = time.time() - infer_time
        infer_times.append(end_infer_time)
        acc_scores.append(accuracy_score(Y_test, Y_pred))
        if classes == 'binary':
            roc_scores.append(roc_auc_score(Y_test, Y_pred))
        else:
            roc_scores.append(roc_auc_score(Y_test, proba, average='macro', multi_class="ovo"))
        matrix = confusion_matrix(Y_test, Y_pred)
        if classes == 'binary':
            tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
            tprs.append(tp / (tp + fn))
            fprs.append(fp / (fp + tn))
        else:
            fp = matrix.sum(axis=0) - np.diag(matrix)
            fn = matrix.sum(axis=1) - np.diag(matrix)
            tp = np.diag(matrix)
            tn = matrix.sum() - (fp + fn + tp)
            fp, fn, tp, tn = fp.astype(float), fn.astype(float), tp.astype(float), tn.astype(float)
            tprs_mc, fprs_mc = [], []
            for i in range(len(fp)):
                tprs_mc.append(tp[i] / (tp[i] + fn[i]))
                fprs_mc.append(fp[i] / (fp[i] + tn[i]))
            tprs.append(mean(tprs_mc))
            fprs.append(mean(fprs_mc))
        precisions.append(precision_score(y_pred=Y_pred, y_true=Y_test, average="macro"))
        auprs_multiclass = []
        for i in range(len(set(Y_test))):
            all_classes_except_i = set(Y_test)
            all_classes_except_i.remove(i)
            binary_y_test = Y_test.replace(i, -1000)
            binary_y_test = binary_y_test.replace(all_classes_except_i, 1)
            binary_y_test = binary_y_test.replace(-1000, 0)
            binary_y_pred = pd.Series(Y_pred).replace(i, -1000)
            binary_y_pred = binary_y_pred.replace(all_classes_except_i, 1)
            binary_y_pred = binary_y_pred.replace(-1000, 0)
            auprs_multiclass.append(average_precision_score(binary_y_test, binary_y_pred))
        auprs.append(sum(auprs_multiclass) / len(auprs_multiclass))

    print("################################################")
    print("Final Scores: ")
    print('Best estimator AUPR:', mean(auprs))
    print('Best estimator acc:', mean(acc_scores))
    print('Best estimator roc:', mean(roc_scores))
    print('Best estimator TPR:', mean(tprs))
    print('Best estimator FPR:', mean(fprs))

    infer_times = [time * 1000 / len(X_test) for time in infer_times]
    data = {
        "Dataset": [dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset],
        "Algorithm Name": ['RandomForest', 'RandomForest', 'RandomForest', 'RandomForest', 'RandomForest',
                           'RandomForest', 'RandomForest', 'RandomForest', 'RandomForest', 'RandomForest'],
        "CV": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "hyperparameter": best,
        "Accuracy": acc_scores,
        "TPR": tprs,
        "FPR": fprs,
        "Precision": precisions,
        "AUC": roc_scores,
        "PR-curves": auprs,
        "Training Time": train_times,
        "Inference Time": infer_times
    }
    pd.DataFrame(data).to_excel('20.xlsx')

    return best_estimator

#Parameters you might want to change for different datasets
dataset_name = 'baseball'
DATASET_PATH = f'C:/Users/barve/PycharmProjects/RFCompare/data/{dataset_name}.csv'
dataset = pd.read_csv(DATASET_PATH)

dataset = dataset.sample(frac=1).reset_index(drop=True)
X = dataset.drop(dataset.columns[-1], axis=1)
Y = dataset[dataset.columns[-1]]


lb = LabelEncoder()
Y = pd.Series(lb.fit_transform(Y))

for column in X.columns:
    X[column] = pd.Series(lb.fit_transform(X[column]))

clf = RandomForestClassifier()
params_grid = [{'n_estimators': [10, 20, 30] + list(range(40, 100, 5)),
                'max_depth': [2, 4, 8, 16, 32, 64]}]

rf_model = perform_nested_cv(clf, X, Y, dataset_name, 'm', params_grid)
