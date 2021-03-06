# Text-Clustering:文本聚类

[![](https://img.shields.io/badge/Python-3.5-blue.svg)](https://www.python.org/)<br>
[![](https://img.shields.io/badge/pandas-0.21.0-brightgreen.svg)](https://pypi.python.org/pypi/pandas/0.21.0)
[![](https://img.shields.io/badge/numpy-1.13.1-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.13.1)
[![](https://img.shields.io/badge/matplotlib-2.1.0-brightgreen.svg)](https://pypi.python.org/pypi/matplotlib/2.1.0)
[![](https://img.shields.io/badge/jieba-0.39-brightgreen.svg)](https://pypi.python.org/pypi/jieba/0.39)
[![](https://img.shields.io/badge/gensim-3.2.0-brightgreen.svg)](https://pypi.python.org/pypi/gensim/3.2.0)
[![](https://img.shields.io/badge/scikit--learn-0.19.1-brightgreen.svg)](https://pypi.python.org/pypi/scikit-learn/0.19.1)

## 语言
Python3.5<br>
## 依赖库
pandas=0.21.0<br>
numpy=1.13.1<br>
matplotlib=2.1.0<br>
jieba=0.39<br>
gensim=3.2.0<br>
scikit-learn=0.19.1<br>


## 项目介绍
* 利用非监督学习的方法进行文本聚类及可视化，从而实现探索功能，是对上一个项目Sentiment-analysis的补充。<br>
* TextClustering文件夹可以直接作为模块使用。

## 用法简介

* ### 导入模块，创建模型
``` python
from TextClustering.TextClustering import TextClustering
model = TextClustering(texts=texts)
```

* ### 通过结巴分词并计算词频
``` python
# stopwords是一个列表，也可以stopwords_path作为停用词的路径（.txt文档，每个停用词一行），加入停用词将会增加计算时间
model.text_cut(stopwords=[' ','(',')'])
# 分词后的结果
model.texts_cut
# 词频
model.word_freq
```

* ### 调用gensim模块创建词向量词包
``` python
# 如果要计算近似词语、用词向量矩阵聚类等，必须先做这步
creat_vocab(sg=0,
            size=5,
            window=5,
            min_count=1,
            vocab_savepath='/models/vocab_word2vec.model')
# 也可以导入
model.load_vocab_word2vec('/TextClustering/models/vocab_word2vec.model')
# 词向量模型
model.vocab_word2vec
```

* ### 构建高频词的维度矩阵
``` python
# 用词向量
model.word2matrix(method='vector', top=200)
# 用词频
model.word2matrix(method='frequency', top=200)
高频词的关联词语
model.words_similar
# 用于聚类的矩阵，共现矩阵或者词向量矩阵
model.word_matrix
# 高频词
model.word_top
# 按词频排序的全部词语
model.word_sequence
```

* ### 计算高频词的关联词语
``` python
# 依赖于词向量，在creat_vocab或load_vocab_word2vec后执行
model.words_similar_cal(n=10)
# 高频词的相似词语
model.words_similar
```

* ### PCA降至二维
``` python
# 通过PCA降至二维，用于可视化，建议后续的聚类采用降维后的数据保持统一
model.decomposition()
# 降维后的数据
model.decomposition_data
# pca模型
model.pca
```

* ### 通过scikit-learn进行非监督学习
``` python
model.clustering(X=model.word_matrix, model_name='KMeans', n_clusters=3)
# 聚类结果
model.labels
# 聚类模型
model.model
```

* ### 计算聚类中心关联词语
``` python
model.get_cluster_similar_words()
# 关联词语
model.cluster_similar_words
```

* ### 可视化聚类结果
``` python
# 需要背景色，会对背景采用同样的模型聚类，展现聚类区间
# color不给就会采用随机数从系统中选择，pixel为背景像素距离，越小越稠密计算时间越长，size为像素点大小，越大越稠密
model.show_decomposition(style='italic',
                         background=True,
                         pixel=0.01,
                         size=20,
                         colors=['red', 'blue', 'brightgreen'],
                         textsize=20,
                         savepath=DIR + '/TextClustering/picture/try1.png',
                         show=True)

# 不需要背景色
model.show_decomposition(style='italic',
                         background=False,
                         colors=['red', 'blue', 'brightgreen'],
                         textsize=20,
                         savepath=DIR + '/TextClustering/picture/try2.png',
                         show=True)
```

## 一个简单的demo供参考，详细的demo请参考demo文件夹内容
### 代码
``` python
from TextClustering.TextClustering import TextClustering
import pandas as pd
import os

DIR = os.path.dirname(__file__)

texts = ['涛哥喜欢吃苹果',
         '涛哥讨厌吃苹果',
         '涛哥非常喜欢吃苹果',
         '涛哥非常讨厌吃苹果']  

# creat model
model = TextClustering(texts=texts)

# cut sentences
model.text_cut(wordlen_min=2, count_method='word')

# load word2vec
model.load_vocab_word2vec(DIR + '/TextClustering/models/vocab_word2vec.model')

# creat wordmatrix
model.word2matrix(method='frequency', top=200)
model.words_similar_cal(n=10)
print('word_matrix:\n',
      pd.DataFrame(model.word_matrix,
                   columns=model.word_top,
                   index=model.word_top))

# use KMeans to cluster
model.decomposition()
model.clustering(X=model.decomposition_data,
                 model_name='KMeans',
                 n_clusters=3)

print('word clustering:\n',
      pd.DataFrame({'word_top': model.word_top,
                    'labels': model.labels},
                   columns=['word_top', 'labels']))

# get cluster image
model.show_decomposition(style='italic',
                         background=True,
                         pixel=0.01,
                         size=20,
                         colors=['red', 'blue', 'brightgreen'],
                         textsize=20,
                         savepath=DIR + '/TextClustering/picture/try1.png',
                         show=True)


model.show_decomposition(style='italic',
                         background=False,
                         pixel=0.01,
                         size=20,
                         colors=['red', 'blue', 'brightgreen'],
                         textsize=20,
                         savepath=DIR + '/TextClustering/picture/try2.png',
                         show=True)
```
![result](https://github.com/renjunxiang/Text-Clustering/blob/master/picture/result.png)<br>

### 可视化结果<br>

* __带背景色__<br>
![try1](https://github.com/renjunxiang/Text-Clustering/blob/master/picture/try1.png)<br>

* __不带背景色__<br>
![try2](https://github.com/renjunxiang/Text-Clustering/blob/master/picture/try2.png)<br>

## 详细demo的部分可视化结果<br>
* __聚类数=3__<br>
![3](https://github.com/renjunxiang/Text-Clustering/blob/master/demo/picture/cluster_3.png)<br>
* __聚类数=10__<br>
![10](https://github.com/renjunxiang/Text-Clustering/blob/master/demo/picture/cluster_10.png)<br>
* __聚类数=20__<br>
![20](https://github.com/renjunxiang/Text-Clustering/blob/master/demo/picture/cluster_20.png)<br>










