from TextClustering.TextClustering import TextClustering
import pandas as pd
import os
import numpy as np
from selenium import webdriver
import time

DIR = os.path.dirname(__file__)

# DIR = 'D:\\github\\Text-Clustering\\demo'

# # 从某小说网站爬部分数据
# profile = webdriver.FirefoxProfile()
# profile.set_preference('permissions.default.image', 2)  # 无图模式
# profile.set_preference('browser.migration.version', 9001)  # 部分需要加上这个
#
# driver = webdriver.Firefox(firefox_profile=profile)
# driver.get("http://www.xqishu.com/dushi/yineng/index.html")
#
# infoes = []
# describtions = []
# titles = []
# urls = []
# for i in range(0, 200):
#     onepage_title = driver.find_elements_by_xpath("//div[@class='listBox']/ul/li/a")
#     titles += [j.text for j in onepage_title]
#
#     onepage_url = driver.find_elements_by_xpath("//div[@class='listBox']/ul/li/a")
#     urls += [j.get_attribute('href') for j in onepage_title]
#
#     onepage_info = driver.find_elements_by_xpath("//div[@class='s']")  # 基本信息
#     info = [[k.split('：')[1] for k in np.array(j.text.split('\n'))[[0, 1, 3]]] for j in onepage_info]
#     infoes += info
#
#     onepage_describtion = driver.find_elements_by_xpath("//div[@class='u']")  # 内容简介
#     describtions += [j.text for j in onepage_describtion]
#     driver.find_elements_by_xpath("//div[@class='tspage']/a[text()='下一页']")[0].click()
# driver.close()
#
# titles = pd.DataFrame(titles, columns=['title'])
# urls = pd.DataFrame(urls, columns=['url'])
# info_all = pd.DataFrame(infoes, columns=['author', 'size', 'update'])
# info_all['summary'] = describtions
# info_all = pd.concat([titles, urls, info_all], axis=1)
# info_all.to_excel(DIR + '/data/raw.xlsx', index=False)
##############################################################################################################
# read data
data = pd.read_excel(DIR + '/data/raw.xlsx')

texts = []
for i in data['summary']:
    if i is np.nan:
        texts.append('')
    else:
        texts.append(i)
##############################################################################################################
# creat model
model = TextClustering(texts=texts)

# cut sentences
model.text_cut(stopwords_path=DIR + '/stopwords.txt')

# creat word2vec(different data should creat its own word2vec)
# model.creat_vocab(sg=0,
#                   size=50,
#                   window=5,
#                   min_count=1,
#                   vocab_savepath=DIR + '/models/vocab_word2vec.model')

model.load_vocab_word2vec(DIR + '/models/vocab_word2vec.model')

# creat wordmatrix
model.word2matrix(method='vector', top=200, similar_n=10)
similar_matrix = model.words_similar

# use pca to decomposition
model.decomposition()
print('前两个成分的特征占比:', model.pca.explained_variance_ratio_[0:2].sum())

result = pd.DataFrame({'word': model.word_top})
word_top_freq = [model.word_freq[i] for i in model.word_top]
result['freq'] = word_top_freq

writer = pd.ExcelWriter(DIR + '/data/result_stopwords.xlsx')
for i in range(3, 21):
    # use KMeans to cluster
    model.clustering(X=model.decomposition_data, model_name='KMeans', n_clusters=i)
    result['cluster_%d' % i] = model.labels
    model.get_cluster_similar_words()
    model.cluster_similar_words.to_excel(writer, sheet_name='cluster_%d' % i, index=True)
    model.show_decomposition(style='italic',
                             background=True,
                             pixel=None,
                             size=20,
                             textsize=20,
                             colors=None,
                             savepath=DIR + '/picture/cluster_%d.png' % i,
                             show=False)
    print('finish:%d' % i)
similar_matrix.to_excel(writer, sheet_name='similar', index=False)
result.to_excel(writer, sheet_name='cluster_result', index=False)
writer.save()
