import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from TextClustering.models.colors import colors_all

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')


def _show_decomposition(decomposition_data=None, label=None, classify=None,
                        background=False, cluster=None, pixel=2,size=20,
                        colors=None, textsize=20, style='italic',
                        savepath=None,show=False):
    '''
    
    :param decomposition_data: 降维后数据
    :param label: 文本
    :param classify: 聚类后标签
    :param background: 是否需要背景色
    :param cluster: 聚类的模型
    :param pixel: 像素间距,越小越密,但增加时间
    :param size: 像素大小,越大越密,在pixel大的时候可以增大size
    :param colors: 染色
    :param textsize: 文本大小
    :param style: 
    :return: 
    '''
    if colors is None:
        label_n = len(np.unique(label))
        colors = random.sample(colors_all.keys(), label_n)
    x = np.array([i[0] for i in decomposition_data])
    y = np.array([i[1] for i in decomposition_data])
    if background == False:
        plt.scatter(x=x, y=y, alpha=0)
        for i in range(len(decomposition_data)):
            plt.text(x[i], y[i], label[i],
                     family='serif', style=style, ha='right', wrap=True,
                     color=colors[classify[i]], size=textsize,
                     fontproperties=zhfont1)
    else:
        cluster = cluster
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1

        '''
        x_min, x_max = decomposition_data[:, 0].min() - 1, decomposition_data[:, 0].max() + 1
        y_min, y_max = decomposition_data[:, 1].min() - 1, decomposition_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = background_label.reshape(xx.shape)
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')
        '''
        xx = np.arange(x_min, x_max, pixel)
        yy = np.arange(y_min, y_max, pixel)

        background_x = np.array(list(xx) * len(yy))
        background_y = np.array(list(yy) * len(xx))
        background = [[background_x[i], background_y[i]] for i in range(len(background_x))]
        background_label = cluster.predict(background)

        for n, color in enumerate(colors):
            plt.scatter(background_x[background_label == n],
                     background_y[background_label == n],
                     color=color,s=size)

        for i in range(len(decomposition_data)):
            plt.text(x[i], y[i], label[i],
                     family='serif', style=style, ha='right', wrap=True,
                     color='black', size=textsize,
                     fontproperties=zhfont1)
    plt.savefig(savepath)
    if show==True:
        plt.show()
