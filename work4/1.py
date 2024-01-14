import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

# 数据集处理
data_path = "./datasets/5f9ae242cae5285cd734b91e-momodel/sms_pub.csv"
data = pd.read_csv(data_path, encoding='utf-8')
data_1 = data[(data['label'] == 1)].sample(frac=1.0)
data_0 = data[(data['label'] == 0)].sample(frac=1.0)[: int(1*len(data_1))]
data_new = pd.concat([data_1, data_0], axis=0).sample(frac=1.0)
# 构建训练集和测试集
X = np.array(data_new.msg_new)
y = np.array(data_new.label)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9,
test_size=0.2)
# 构建分类器
# 朴素贝叶斯
estimator = MultinomialNB()
#estimator = ComplementNB()
# # 决策树算法
#estimator = DecisionTreeClassifier(criterion="entropy")
# # 随机森林算法
# # 1.随即森林预估器
##estimator =RandomForestClassifier()
# # 2.参数准备
#param_dict = {"n_estimators": [90,100,110], "max_depth":[100,200,300]}
#estimator = GridSearchCV(estimator, param_dict, cv=3)
# # 支持向量机
estimator=svm.SVC(kernel='rbf',C=1000, probability=True)
# 2.KNN算法预估器
estimator = KNeighborsClassifier()

pipeline = Pipeline([
('tfidf', TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",
stop_words=stopwords, ngram_range=(1,2))),
('MaxAbsScaler', MaxAbsScaler()),
('classifier', estimator)
])
# 模型训练
pipeline.fit(X_train, y_train)
# 模型评价
y_pred = pipeline.predict(X_test)
print("在测试集上的混淆矩阵：")
print(metrics.confusion_matrix(y_test, y_pred))
print("在测试集上的分类结果报告：")
print(metrics.classification_report(y_test, y_pred))
print("在测试集上的 f1-score ：")
print(metrics.f1_score(y_test, y_pred))
print("在测试集上的 AUC为:")
print(roc_auc_score(y_test, y_pred))
print('在测试集上的准确率：')
print(metrics.accuracy_score(y_test, y_pred))

# 保存模型
pipeline.fit(X, y)
joblib.dump(pipeline, 'results/pipeline.model')