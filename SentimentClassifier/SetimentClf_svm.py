from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC

# 加载文件，导入数据,分词
def loadfile():
    neg = pd.read_excel('datasets/neg.xls', header=None, index=None)
    pos = pd.read_excel('datasets/pos.xls', header=None, index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    # print pos['words']
    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

    np.save('svm_data/y_train.npy', y_train)
    np.save('svm_data/y_test.npy', y_test)
    return x_train, x_test

#对每个句子的所有词向量取均值
def buildWordVector(text, size,model_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def get_train_vecs(x_train,x_test):
    n_dim = 100
    train_model = Word2Vec(x_train,size=n_dim,window=5,workers=5)
    train_vecs = np.concatenate([buildWordVector(z, n_dim, train_model) for z in x_train])

    # train_vecs = scale(train_vecs)
    np.save('svm_data/train_vecs.npy', train_vecs)
    print (train_vecs.shape)
    # Train word2vec on test tweets
    test_model = Word2Vec(x_test,size=n_dim,window=5,workers=5)
    test_model.save('svm_data/test_w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim, train_model) for z in x_test])
    # test_vecs = scale(test_vecs)
    np.save('svm_data/test_vecs.npy', test_vecs)
    print(test_vecs.shape)

def get_data():
    train_vecs = np.load('svm_data/train_vecs.npy')
    y_train = np.load('svm_data/y_train.npy')
    test_vecs = np.load('svm_data/test_vecs.npy')
    y_test = np.load('svm_data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test


##训练svm模型
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, 'svm_data/svm_model.pkl')
    print('SVM.clf.Score==',clf.score(test_vecs, y_test))

##得到待预测单个句子的词向量
def get_predict_vecs(words):
    n_dim = 100
    w2v_model = Word2Vec.load('svm_data/test_w2v_model.pkl')
    # imdb_w2v.train(words)
    predict_vecs = buildWordVector(words, n_dim, w2v_model)
    # print train_vecs.shape
    return predict_vecs


####对单个句子进行情感判断
def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('svm_data/svm_model.pkl')

    result = clf.predict(words_vecs)

    if int(result[0]) == 1:
        print(string, ' positive')
    else:
        print(string, ' negative')

if __name__ == '__main__':
    # 得到句子分词后的结果，并把类别标签保存为y_train.npy,y_test.npy
    x_train,x_test=loadfile()
    # 计算词向量并保存为train_vecs.npy,test_vecs.npy
    get_train_vecs(x_train,x_test)
    # 导入训练数据和测试数据
    train_vecs,y_train,test_vecs,y_test=get_data()
    #训练svm并保存模型
    svm_train(train_vecs,y_train,test_vecs,y_test)

##对输入句子情感进行判断
string1='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
string2='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
string3='酒店的环境非常好，价格也便宜，值得推荐'
string4='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
string5='你是傻逼'
string6='屏幕较差，拍照也很粗糙。'
string7='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
string8='东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'
svm_predict(string1)
svm_predict(string2)
svm_predict(string3)
svm_predict(string4)
svm_predict(string5)
svm_predict(string6)
svm_predict(string7)
svm_predict(string8)
