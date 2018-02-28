import pandas as pd
import numpy as np
import os
import jieba
from collections import defaultdict
from sklearn.decomposition import PCA
from gensim.models import word2vec

from TextClustering.transform.creat_vocab_word2vec import creat_vocab_word2vec
from TextClustering.models.clustering import clustering
from TextClustering.models.show_decomposition import _show_decomposition

jieba.setLogLevel('WARN')

DIR = os.path.dirname(__file__)


class TextClustering():
    def __init__(self,
                 texts=None):
        self.texts = texts

    def text_cut(self, stopwords_path=None, stopwords=None,
                 wordlen_min=2, count_method='word'):
        '''
        
        :param stopwords_path: 停用词路径
        :param stopwords: 停用词列表，高优先级
        :param wordlen_min: 词语最短长度
        :param count_method: 词频计算方法，'word'词语出现的数量类似TF，'text'包含这个词语的文本数量类似IDF
        :return: 
        '''
        if stopwords_path == 'default':
            stopwords_path = DIR + '/transform/stopwords.txt'
        if stopwords is None and stopwords_path is not None:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stopwords = f.read().splitlines()
        texts = self.texts
        self.stopwords = stopwords

        word_freq_full = defaultdict(int)
        texts_cut = []
        for one_text in texts:
            text_cut = [word for word in jieba.lcut(one_text) if word != ' ']  # 每句话分词
            texts_cut.append(text_cut)
            # if stopwords is not None:
            #     for word in text_cut:
            #         if word not in stopwords:
            #             word_freq_full[word] += 1  # 去除停用词，计算词频
            # else:
            #     for word in text_cut:
            #         word_freq_full[word] += 1  # 计算词频
            if count_method == 'word':
                for word in text_cut:
                    word_freq_full[word] += 1  # 计算词语出现数量
            else:
                for word in set(text_cut):
                    word_freq_full[word] += 1  # 计算包含词语的文本数量
        # 之前方法判断次数过多，采用下面的方式仅判断一次
        if stopwords is not None:  # 有停用词则剔除其词频
            word_freq = {}
            for word in word_freq_full:
                if word not in stopwords and len(word) >= wordlen_min:
                    word_freq.update({word: word_freq_full[word]})
        else:
            word_freq = {}
            for word in word_freq_full:
                if len(word) >= wordlen_min:
                    word_freq.update({word: word_freq_full[word]})

        self.texts_cut = texts_cut
        self.word_freq = word_freq

    def creat_vocab(self,
                    sg=0,
                    size=5,
                    window=5,
                    min_count=1,
                    vocab_savepath=DIR + '/models/vocab_word2vec.model'):
        '''
        get dictionary by word2vec
        :param texts: list of text
        :param sg: 0 CBOW,1 skip-gram
        :param size: the dimensionality of the feature vectors
        :param window: the maximum distance between the current and predicted word within a sentence
        :param min_count: ignore all words with total frequency lower than this
        :param vocab_savepath: path to save word2vec dictionary
        :return: None
        '''
        # 构建词向量词库
        texts = self.texts
        self.vocab_word2vec = creat_vocab_word2vec(texts=texts,
                                                   sg=sg,
                                                   vocab_savepath=vocab_savepath,
                                                   size=size,
                                                   window=window,
                                                   min_count=min_count)

    def load_vocab_word2vec(self,
                            vocab_loadpath=DIR + '/models/vocab_word2vec.model'):
        '''
        load dictionary
        :param vocab_loadpath: path to load word2vec dictionary
        :return: 
        '''
        self.vocab_word2vec = word2vec.Word2Vec.load(vocab_loadpath)

    def word2matrix(self, method='vector', top=50):
        '''
        根据文本生成数据集词库的矩阵供聚类使用
        :param method: frequency or vector
        :param top: number of freq words
        :return: 
        '''
        texts_cut = self.texts_cut
        word_freq = self.word_freq
        if len(word_freq) < top:
            top = len(word_freq)
        word_sequence = sorted(word_freq, key=lambda x: word_freq[x], reverse=True)
        word_top = word_sequence[0:top]
        if method == 'frequency':
            word_matrix = np.zeros(shape=[top, top])
            # for row_index in range(top):
            #     for col_index in range(top):
            #         for one_text_cut in texts_cut:
            #             if (word_top[row_index] in one_text_cut) and (word_top[col_index] in one_text_cut):
            #                 word_matrix[row_index, col_index] += 1
            # 采用矢量化运算代替循环，速度大幅提高
            for one_text_cut in texts_cut:
                word_matrix1d = np.matrix(np.in1d(word_top, one_text_cut))
                word_matrix += word_matrix1d.T * word_matrix1d
        elif method == 'vector':
            vocab_word2vec = self.vocab_word2vec
            word_matrix = np.array([vocab_word2vec[i] for i in word_top])
        self.word_matrix = word_matrix
        self.word_top = word_top
        self.word_sequence = word_sequence

    def words_similar_cal(self, n=10):
        '''
        计算高频词的相关词语
        :param n: number of similar words
        :return: 
        '''
        word_top = self.word_top
        vocab_word2vec = self.vocab_word2vec
        words_similar = []
        for word in word_top:
            word_similar = vocab_word2vec.wv.most_similar(word, topn=n)
            word_similar = [[word] + list(one_similar) for one_similar in word_similar]
            words_similar += word_similar
        words_similar = pd.DataFrame(words_similar, columns=['word', 'similar', 'score'])
        self.words_similar = words_similar

    def decomposition(self, n_components=2):
        X = self.word_matrix
        model = PCA(n_components=n_components)
        decomposition_data = model.fit_transform(X=X)
        self.pca = model
        self.decomposition_data = decomposition_data

    def clustering(self, X=None, model_name='KMeans', n_clusters=3):
        word_top = self.word_top
        word_top_num = len(word_top)
        if n_clusters > word_top_num:
            n_clusters = len(word_top)
            print("'n_clusters' is larger than words' number,set n_clusters=%d" % (word_top_num))
        self.train_data = X
        self.model, self.labels = clustering(X=X,
                                             model_name=model_name,
                                             n_clusters=n_clusters)

    def get_cluster_similar_words(self):
        words_similar = self.words_similar
        labels = self.labels
        word_top = self.word_top

        cluster_result = pd.DataFrame({'word': word_top, 'labels': labels})
        cluster_result_similar = pd.merge(cluster_result,
                                          words_similar.iloc[:, 0:2],
                                          left_on='word', right_on='word')
        group = cluster_result_similar.groupby('labels')
        cluster_similar_words = group['similar'].value_counts()
        self.cluster_similar_words = pd.DataFrame(cluster_similar_words)

    def show_decomposition(self,
                           background=False,
                           pixel=None,
                           size=20,
                           colors=None,
                           textsize=20,
                           style='italic',
                           savepath=None,
                           show=False):
        '''
        only KMeans can show with background
        :param background: 是否需要背景色
        :param cluster: 聚类的模型
        :param pixel: 像素间距,越小越密,但增加时间
        :param size: 像素大小,越大越密,在pixel大的时候可以增大size
        :param colors: 染色
        :param textsize: 文本大小
        :return: 
        '''
        model = self.model
        decomposition_data = self.decomposition_data

        _show_decomposition(background=background,
                            cluster=model,
                            decomposition_data=decomposition_data,
                            words=self.word_top,
                            classify=self.labels,
                            pixel=pixel,
                            size=size,
                            colors=colors,
                            textsize=textsize,
                            style=style,
                            savepath=savepath,
                            show=show)
