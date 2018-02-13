# Text-Clustering

## 语言
Python3.5<br>
## 依赖库
requests=2.18.4<br>
pandas=0.21.0<br>
numpy=1.13.1<br>
matplotlib=2.1.0<br>
jieba=0.39<br>
gensim=3.2.0<br>
scikit-learn=0.19.1<br>


## 项目介绍
利用非监督学习的方法进行文本聚类及可视化，从而实现探索功能，是对上一个项目Sentiment-analysis的补充。TextClustering已封装为模块，可以直接使用。

## 用法简介
该模块包含：<br>
1.通过结巴分词，计算词频；<br>
2.计算共现矩阵<br>
3.通过gensim模块创建词向量词包<br>
4.构建高频词的维度矩阵<br>
5.通过scikit-learn进行非监督学习<br>
6.计算聚类中心关联词语<br>
7.可视化聚类结果<br>

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
model.text_cut()

# load word2vec
model.load_vocab_word2vec(DIR + '/TextClustering/models/vocab_word2vec.model')

# creat wordmatrix
model.word2matrix(method='frequency', top=200)
print('similiar matrix:\n',
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
                         colors=['red', 'blue', 'green'],
                         textsize=20,
                         savepath=DIR + '/TextClustering/picture/try1.png',
                         show=True)


model.show_decomposition(style='italic',
                         background=False,
                         pixel=0.01,
                         size=20,
                         colors=['red', 'blue', 'green'],
                         textsize=20,
                         savepath=DIR + '/TextClustering/picture/try2.png',
                         show=True)
```
可视化结果<br>

带背景色<br>
![try1](https://github.com/renjunxiang/Text-Clustering/blob/master/TextClustering/picture/try1.png)<br>

不带背景色<br>
![try2](https://github.com/renjunxiang/Text-Clustering/blob/master/TextClustering/picture/try2.png)<br>














