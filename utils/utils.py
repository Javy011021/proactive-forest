import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, accuracy_score, precision_score


def shuffle_data(data, labels):
    if (len(data) != len(labels)):
        raise Exception(
            "The given data and labels do NOT have the same length")
    lista = []

    for i, j in zip(data, labels):
        lista.append(np.r_[i, [j]])
    lista = np.array(lista)
    np.random.shuffle(lista)
    x, y = [], []
    for i in lista:
        x.append(i[:-1])
        y.append(i[-1])

    return np.array(x), np.array(y)


def create_k(x, y, k=5, type="skf"):
    train = []
    test = []
    fold = None
    # data, labels = shuffle_data(x,y)
    data, labels = x, y
    if type == "kf":
        fold = KFold(n_splits=k)
    elif type == "skf":
        fold = StratifiedKFold(n_splits=k)
    else:
        raise ValueError(
            "The specified cross-validation type was not recognized.")

    for train_index, test_index in fold.split(data, labels):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        train.append([x_train, y_train])
        test.append([x_test, y_test])

    return train, test


def cross_validation_train(model, train, test):
    score_max = 0
    avg_recall = 0
    avg_presi = 0
    avg_auc = 0
    avg_acc = 0
    avg_pcd = 0
    best_model = None
    a = 1
    results = []
    real_y = np.array([], dtype=str)
    predicc_y = np.array([], dtype=str)
    for trainn, testss in zip(train, test):
        x_train = trainn[0]
        x_test = testss[0]

        y_train = trainn[1]
        y_test = testss[1]
        print("Para el", a, " k conjunto de prueba y entrenamiento")

        model.fit(x_train, y_train)
        # score_auc = calculate_roc_auc(np.unique(y_train) ,np.unique(y_test), model, x_test, y_test)
        score_auc = 0
        predictions = model.predict(x_test)
        conf_matrx = confusion_matrix(y_test, predictions)
        # print("para y", np.unique(y_test),"para predict",np.unique(predictions))

        real_y = np.append(real_y, y_test)
        predicc_y = np.append(predicc_y, predictions)

        print(conf_matrx)
        score_recll = recall_score(y_test, predictions, average='macro')
        score_presi = precision_score(y_test, predictions, average='macro')
        score_acc = accuracy_score(y_test, predictions)
        pcd = model.diversity_measure(x_test, y_test)

        avg_recall = score_recll + avg_recall
        avg_auc = score_auc + avg_auc
        avg_presi = score_presi + avg_presi
        avg_acc = score_acc + avg_acc
        avg_pcd = pcd + avg_pcd

        # print("The recall of group", a, "is", score_recll, ", roc_auc is", score_auc,", accuracy is", score_acc, "and diversity PCD is", pcd)
        a += 1

    avg_recall = avg_recall/len(train)
    avg_auc = avg_auc/len(train)
    avg_acc = avg_acc/len(train)
    avg_pcd = avg_pcd/len(train)
    avg_presi = avg_presi/len(train)

    # print("matriz final")
    # print("para y", np.unique(real_y),"para predict",np.unique(predicc_y))
    # print(confusion_matrix(real_y, predicc_y))

    print("The final cross_val recall is", avg_recall, ", roc_auc is", avg_auc,
          ", precision is", avg_presi, ", accuracy is", avg_acc, "and diversity PCD is", avg_pcd)
    # print("Max score is", score_max)
    return avg_recall, avg_auc, avg_acc, avg_pcd, avg_presi


def calculate_roc_auc(y_train_class, y_test_class, model, x_test, y_test):
    indexs = []
    cont_indexs = 0
    not_indexs = []
    for i in y_test_class:
        try:
            indexs.append(np.where(y_train_class == i)[0][0])
        except:
            not_indexs.append(cont_indexs)
        cont_indexs += 1

    proba = np.array(model.predict_proba(x_test, indexs))
    if len(not_indexs) > 0:
        for i in not_indexs:
            proba = np.insert(proba, i, 0, axis=1)

    try:
        score_roc_auc = roc_auc_score(
            y_test, proba, multi_class='ovr', average='macro')
    except:
        score_roc_auc = roc_auc_score(y_test, proba[:, 1])

    return score_roc_auc


# Método para dividir el conjunto de entrenamiento
def train_test_splitt(train_data, train_labels, test_size=0.2, shuffle=False):
    if shuffle:
        train_data, train_labels = shuffle_data(train_data, train_labels)

    split_i = len(train_labels) - int(len(train_labels) // (1 / test_size))
    x_train, x_test = train_data[:split_i], train_data[split_i:]
    y_train, y_test = train_labels[:split_i], train_labels[split_i:]

    return x_train, x_test, y_train, y_test


def cross_validation_train_with_pruning(model, train, test, pruning=None):
    avg_recall = 0
    avg_presi = 0
    avg_auc = 0
    avg_acc = 0
    avg_pcd = 0
    avg_initial_size = 0
    avg_final_size = 0
    avg_time = 0
    a = 1

    for trainn, testss in zip(train, test):
        x_train = trainn[0]
        x_test = testss[0]

        y_train = trainn[1]
        y_test = testss[1]
        print("Para el", a, " k conjunto de prueba y entrenamiento")

        model.fit(x_train, y_train, pruning if pruning ==
                  "window_threshold" else False)

        score_recll, score_auc, score_acc, pcd, score_presi, duration = get_metrics(
            model, x_test, y_test, True if (not pruning) or pruning == "window_threshold" else False)

        if pruning and pruning != "window_threshold":
            model_initial_size, model_final_size = model.pruning(
                x_test, y_test, pruning, accuracy=score_auc)
            score_recll, score_auc, score_acc, pcd, score_presi, _ = get_metrics(
                model, x_test, y_test)
        else:
            model_initial_size = 100
            model_final_size = len(model._trees)
            avg_time = duration + avg_time

        avg_recall = score_recll + avg_recall
        avg_auc = score_auc + avg_auc
        avg_presi = score_presi + avg_presi
        avg_acc = score_acc + avg_acc
        avg_pcd = pcd + avg_pcd
        avg_initial_size = model_initial_size + avg_initial_size
        avg_final_size = model_final_size + avg_final_size

        a += 1

    avg_recall = avg_recall/len(train)
    avg_auc = avg_auc/len(train)
    avg_acc = avg_acc/len(train)
    avg_pcd = avg_pcd/len(train)
    avg_presi = avg_presi/len(train)
    model_initial_size = avg_initial_size/len(train)
    model_final_size = avg_final_size/len(train)

    print("The final cross_val recall is", avg_recall, ", roc_auc is", avg_auc,
          ", precision is", avg_presi, ", accuracy is", avg_acc, "and diversity PCD is", avg_pcd)
    return avg_recall, avg_auc, avg_acc, avg_pcd, avg_presi, model_initial_size, model_final_size, avg_time/len(train)


def get_metrics(model, x_test, y_test, show_matix=True):
    score_auc = 0
    start = time.time()
    predictions = model.predict(x_test)
    if show_matix:
        conf_matrx = confusion_matrix(y_test, predictions)
        print(conf_matrx)

    score_recll = recall_score(y_test, predictions, average='macro')
    score_presi = precision_score(y_test, predictions, average='macro')
    score_acc = accuracy_score(y_test, predictions)
    pcd = model.diversity_measure(x_test, y_test)
    
    end = time.time()
    duration = (end-start) / 60

    return score_recll, score_auc, score_acc, pcd, score_presi, duration




def cross_validation_test(model, train, test, acc_threshold, div_threshold):
    score_max = 0
    avg_recall = 0
    avg_presi = 0
    avg_auc = 0
    avg_acc = 0
    avg_pcd = 0
    best_model = None
    a = 1
    results = []
    real_y = np.array([], dtype=str)
    predicc_y = np.array([], dtype=str)
    for trainn, testss in zip(train, test):
        x_train = trainn[0]
        x_test = testss[0]

        y_train = trainn[1]
        y_test = testss[1]
        print("Para el", a, " k conjunto de prueba y entrenamiento")

        model.test_threshold(x_train, y_train, diversity_threshold=div_threshold, accuracy_threshold=acc_threshold)
        # score_auc = calculate_roc_auc(np.unique(y_train) ,np.unique(y_test), model, x_test, y_test)
        score_auc = 0
        predictions = model.predict(x_test)
        conf_matrx = confusion_matrix(y_test, predictions)
        # print("para y", np.unique(y_test),"para predict",np.unique(predictions))

        real_y = np.append(real_y, y_test)
        predicc_y = np.append(predicc_y, predictions)

        print(conf_matrx)
        score_recll = recall_score(y_test, predictions, average='macro')
        score_presi = precision_score(y_test, predictions, average='macro')
        score_acc = accuracy_score(y_test, predictions)
        pcd = model.diversity_measure(x_test, y_test)

        avg_recall = score_recll + avg_recall
        avg_auc = score_auc + avg_auc
        avg_presi = score_presi + avg_presi
        avg_acc = score_acc + avg_acc
        avg_pcd = pcd + avg_pcd

        # print("The recall of group", a, "is", score_recll, ", roc_auc is", score_auc,", accuracy is", score_acc, "and diversity PCD is", pcd)
        a += 1

    avg_recall = avg_recall/len(train)
    avg_auc = avg_auc/len(train)
    avg_acc = avg_acc/len(train)
    avg_pcd = avg_pcd/len(train)
    avg_presi = avg_presi/len(train)

    # print("matriz final")
    # print("para y", np.unique(real_y),"para predict",np.unique(predicc_y))
    # print(confusion_matrix(real_y, predicc_y))

    print("The final cross_val recall is", avg_recall, ", roc_auc is", avg_auc,
          ", precision is", avg_presi, ", accuracy is", avg_acc, "and diversity PCD is", avg_pcd)
    # print("Max score is", score_max)
    return avg_recall, avg_auc, avg_acc, avg_pcd, avg_presi