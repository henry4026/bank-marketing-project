from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

cost_benefit_1 = np.array([[70, -100], [-30, 0]])
cost_benefit_ = pd.DataFrame([[0,-30],[-100,70]])
cost_benefit_.index.name = 'actual'
cost_benefit_.columns.name = 'predicted'

def confusion_matrix_(model, X,y, threshold=0.5):
    cf = pd.crosstab(y, predict(model, X, threshold))
    cf.index.name = 'actual'
    cf.columns.name = 'predicted'
    return cf

def get_confusion_matrix(model,X,y,threshold):
    tn = confusion_matrix_(model, X,y, threshold=threshold)[0][0]
    fp = confusion_matrix_(model, X,y, threshold=threshold)[1][0]
    fn = confusion_matrix_(model, X,y, threshold=threshold)[0][1]
    tp = confusion_matrix_(model, X,y, threshold=threshold)[1][1]
    print(np.array([[tp, fn], [fp, tn]]))
    print("precision:{precision}, recall:{recall}".format(precision = tp/(tp+fp),recall = tp/(tp+fn)))
    print("total profit:{profit}".format(profit = ((confusion_matrix_(model, X,y, threshold=threshold) * cost_benefit_).values.sum())))
def accuracy_(model,X,y):
    pred = model.predict(X)
    print("accuracy:{accuracy}".format(accuracy=model.score(X, y)))
    print("r2:{r2}".format(r2 = r2_score(y, pred)))
    print("f1:{f1}".format(f1 = f1_score(y,pred)))
def nn_get_confusion_matrix(model,X,y,threshold):
    pred = model.predict_classes(X)
#     tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
#     print(np.array([[tp, fn], [fp, tn]]))
    tn = nn_confusion_matrix_(model, X,y, threshold=threshold)[0][0]
    fp = nn_confusion_matrix_(model, X,y, threshold=threshold)[1][0]
    fn = nn_confusion_matrix_(model, X,y, threshold=threshold)[0][1]
    tp = nn_confusion_matrix_(model, X,y, threshold=threshold)[1][1]
    print(np.array([[tp, fn], [fp, tn]]))
    print("precision:{precision}, recall:{recall}".format(precision = tp/(tp+fp),recall = tp/(tp+fn)))
    print("total profit:{profit}".format(profit = ((nn_confusion_matrix_(model, X,y, threshold=threshold) * cost_benefit_).values.sum())))
# def nn_accuracy_(model,X,y):
#     pred = model.predict_classes(X)
#     print(f"accuracy = {model.score(X, y)}")
#     print(f"r2 = {r2_score(y, pred)}")
#     print(f"f1 = {f1_score(y,pred)}")
def calculate_payout(cost_benefit_, model, X, y,threshold=0.5):
    return (confusion_matrix_(model, X,y, threshold) * cost_benefit_).values.sum()

def predict(model, X, threshold=0.5):
    '''Return prediction of the fitted binary-classifier model model on X using
    the specifed `threshold`. NB: class 0 is the positive class'''
    return np.where(model.predict_proba(X)[:, 0] > threshold,
                    model.classes_[0],
                    model.classes_[1])

def nn_predict(model,X,threshold=0.5):
    length_of_nn = len(model.predict_proba(X,batch_size=16))
    lst = np.zeros([length_of_nn,2])
    result = lst[:,1].reshape(length_of_nn,1)
    result=(model.predict_proba(X,batch_size=32))
    lst[:,1]=result.reshape(length_of_nn)
    lst[:,0]=(1-result.reshape(length_of_nn))
    return np.where(lst[:, 0] > threshold, 0,1)

def nn_confusion_matrix_(model,X,y,threshold=0.5):
    cf = pd.crosstab(y, nn_predict(model, X, threshold))
    cf.index.name = 'actual'
    cf.columns.name = 'predicted'
    return cf

def nn_calculate_payout(cost_benefit_, model, X,y, threshold):
    return (nn_confusion_matrix_(model, X,y, threshold) * cost_benefit_).values.sum()

def profit_curve(rf,gdb_model,logistic_model,xgb_model,nn_model,X,y):
    thresholds = np.arange(0.0, 1.0, 0.05)
    profits = []
    profits1 = []
    profits2 = []
    profits3 = []
    profits4 = []
    for threshold in thresholds:
        profits.append(calculate_payout(cost_benefit_, rf, X,y, threshold))
    for threshold in thresholds:
        profits1.append(calculate_payout(cost_benefit_, gdb_model, X,y, threshold))
    for threshold in thresholds:
        profits2.append(calculate_payout(cost_benefit_, logistic_model, X,y, threshold))
    for threshold in thresholds:
        profits3.append(calculate_payout(cost_benefit_, xgb_model, X,y, threshold))
    for threshold in thresholds:
        profits4.append(nn_calculate_payout(cost_benefit_, nn_model, X,y, threshold))
    fig, ax = plt.subplots()
    ax.plot(thresholds, profits,label = 'random forest',color='r')
    ax.plot(thresholds, profits1,label = 'gradient boost',color='b')
    ax.plot(thresholds, profits2,label='logistic',color='g')
    ax.plot(thresholds, profits3,label='xgb',color='k')
    ax.plot(thresholds, profits4,label='nn',color='y')
    ax.legend()
    ax.set_xlabel('thresholds')
    ax.set_ylabel('profits')
    ax.set_title('Profit Curve')
    plt.show()
    
def feat_imp(df, model, n_features):

    d = dict(zip(df.columns, model.feature_importances_))
    ss = sorted(d, key=d.get, reverse=True)
    top_names = ss[0:n_features]
    lst=[]
    for i in ss:
        lst.append(d[i])
    plt.figure(figsize=(15,15))
    plt.title("Feature Importances",fontsize=28)
    plt.barh(range(n_features),(sorted([d[i] for i in top_names],reverse=False)), color="b", align="center")
#     plt.ylim(-1, n_features)
    plt.yticks(range(n_features), top_names[::-1],fontsize=20)
#     ax.invert_yaxis()
#     ax.tick_params(axis='both', which='major', labelsize=15, color='white')

def find_best_profit(model,X,y):
    threshold_lst = []
    values_lst = []
    th = np.arange(0.2, 1.1, 0.05)
    for i in th:
        values_lst.append(((confusion_matrix_(model,X,y,threshold=i) * cost_benefit_).values.sum()))
        threshold_lst.append(i)
    profit_result = max(values_lst)
    threshold_result = (threshold_lst[values_lst.index(profit_result)])
    return (profit_result, threshold_result)

def find_best_profit_nn(model,X,y):
    threshold_lst = []
    values_lst = []
    th = np.arange(0.1, 1.1, 0.05)
    for i in th:
        values_lst.append(((nn_confusion_matrix_(model,X,y,threshold=i) * cost_benefit_).values.sum()))
        threshold_lst.append(i)
    profit_result = max(values_lst)
    threshold_result = (threshold_lst[values_lst.index(profit_result)])
    return (profit_result, threshold_result)

def roc_curve_(rf,gdb_model,logistic_model,xgb_model,nn_model,X_test,y_test):
    probs = rf.predict_proba(X_test)
    preds = probs[:,1]
    probs1 = gdb_model.predict_proba(X_test)
    preds1 = probs1[:,1]
    probs2 = logistic_model.predict_proba(X_test)
    preds2 = probs2[:,1]
    probs3 = xgb_model.predict_proba(X_test)
    preds3 = probs3[:,1]
    prob4 = nn_model.predict_proba(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, preds1)
    fpr2, tpr2, threshold2 = metrics.roc_curve(y_test, preds2)
    fpr3, tpr3, threshold3 = metrics.roc_curve(y_test, preds3)
    fpr4, tpr4, threshold4 = metrics.roc_curve(y_test, prob4)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    roc_auc3 = metrics.auc(fpr3, tpr3)
    roc_auc4 = metrics.auc(fpr4,tpr4)
    plt.title('ROC curve')
    plt.plot(fpr, tpr, 'b', label = "RF AUC = {0:.1%}".format(round(roc_auc,3)))
    plt.plot(fpr1, tpr1, 'r', label = "GB AUC = {0:.1%}".format(round(roc_auc1,3)))
    plt.plot(fpr2, tpr2, 'g', label = "logistic AUC = {0:.1%}".format(round(roc_auc2,3)))
    plt.plot(fpr3, tpr3, 'k', label = "xgboost AUC = {0:.1%}".format(round(roc_auc3,3)))
    plt.plot(fpr4, tpr4, 'y', label = "neural network AUC = {0:.1%}".format(round(roc_auc4,3)))
    plt.legend(loc = 'lower right',shadow=True)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()