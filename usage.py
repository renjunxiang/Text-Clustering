from TextClustering.TextClustering import TextClustering
import pandas as pd
import os

DIR = os.path.dirname(__file__)

texts = ['涛哥喜欢吃苹果',
         '涛哥讨厌吃苹果',
         '涛哥非常喜欢吃苹果',
         '涛哥非常讨厌吃苹果']  # creat model
model = TextClustering(texts=texts)

# cut sentences
model.text_cut(wordlen_min=2, count_method='word')

# creat word2vec
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
                         # savepath=DIR + '/picture/try1.png',
                         show=True)


model.show_decomposition(style='italic',
                         background=False,
                         pixel=0.01,
                         size=20,
                         colors=['red', 'blue', 'green'],
                         textsize=20,
                         # savepath=DIR + '/picture/try2.png',
                         show=True)
