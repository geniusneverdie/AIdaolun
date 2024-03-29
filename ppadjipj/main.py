import os
import warnings

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler

# 数据集的路径
data_path = "./datasets/5f9ae242cae5285cd734b91e-momodel/sms_pub.csv"
# 读取数据
sms = pd.read_csv(data_path, encoding='utf-8')
pos = sms[(sms['label'] == 1)]
neg = sms[(sms['label'] == 0)].sample(frac=1.0)[: len(pos)]
sms = pd.concat([pos, neg], axis=0).sample(frac=1.0)

# ---------- 停用词库路径，若有变化请修改 -------------
stopwords_path = r'scu_stopwords.txt'

# ---------------------------------------------------


"""
读取停用词库
:param stopwords_path: 停用词库的路径
:return: 停用词列表，如 ['嘿', '很', '乎', '会', '或']
"""


def read_stopwords(stopwords_path):
    stopwords = []
# ----------- 请完成读取停用词的代码 ------------
    with open(stopwords_path, encoding='utf-8-sig'):
        lines = stopwords_path.readlines()
    for line in lines:
       stopwords.append(line.rstrip())
# ----------------------------------------------
    return stopwords

# 读取停用词
stopwords = read_stopwords(stopwords_path)

# ----------------- 导入相关的库 -----------------
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.externals import joblib

# 构建和划分训练集和测试集
X = np.array(sms.msg_new)
Y = np.array(sms.label)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.1)

# pipline_list用于传给Pipline作为参数
pipeline_list = [
    # --------------------------- 需要完成的代码 ------------------------------
    # ========================== 以下代码仅供参考 =============================
    ('Tfid', TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
    ('classifier', MultinomialNB())
    # ========================================================================
    # ------------------------------------------------------------------------
]

# 搭建 pipeline
pipeline = Pipeline(pipeline_list)
# 训练 pipeline
pipeline.fit(X_train, y_train)
# 在所有的样本上训练一次，充分利用已有的数据，提高模型的泛化能力
pipeline.fit(X, Y)

# 保存训练的模型，请将模型保存在 results 目录下
pipeline_path = 'results/pipeline.model'
joblib.dump(pipeline, pipeline_path)

# 测试
y_pred = pipeline.predict(X_test)
print("在测试集上的 f1-score ：")
print(metrics.f1_score(y_test, y_pred))
print('在测试集上的准确率：')
print(metrics.accuracy_score(y_test, y_pred))

# 加载训练好的模型
from sklearn.externals import joblib

# ------- pipeline 保存的路径，若有变化请修改 --------
pipeline_path = 'results/pipeline.model'
# --------------------------------------------------
pipeline = joblib.load(pipeline_path)

"""
预测短信短信的类别和每个类别的概率
param: message: 经过jieba分词的短信，如"医生 拿 着 我 的 报告单 说 ： 幸亏 你 来 的 早 啊"
return: label: 整数类型，短信的类别，0 代表正常，1 代表恶意
proba: 列表类型，短信属于每个类别的概率，如[0.3, 0.7]，认为短信属于 0 的概率为 0.3，属于 1 的概率为 0.7
"""


def predict(message):
    label = pipeline.predict([message])[0]
    proba = list(pipeline.predict_proba([message])[0])
    return label, proba
if __name__ == '__main__':
    import pandas as pd
import numpy as np

data_eval = pd.read_csv('sms_eval.csv', encoding='utf8')
y_eval = np.array(data_eval['label'])
X_eval = np.array(data_eval['msg_new'])
total = y_eval.shape[0]
count = 0
for x, y in zip(X_eval, y_eval):
    y_pred, _ = predict(x)
if y_pred == y:
    count += 1
print('{} / {}'.format(count, total))
