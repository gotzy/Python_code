
from gensim.models import word2vec
 
#open file patten 01
word_list=[ x.rstrip("\r\n").split() for x in list(open("infile")) ]
 
#open file patten 02
#word_list = word2vec.LineSentence("infile")
 
#modelling
model = word2vec.Word2Vec(word_list, size=100,min_count=1,window=5,iter=100)
 
#save model
model.save("filename.model")
 
# save as vector（１行目は単語数とベクトルの次元の情報）
model.wv.save_word2vec_format("filename.vec",binary=False)
 
#list of words which were vectorized
list_words = model.wv.index2word
 
# word vector of "hoge"
model.wv.word_vec('hoge')
 
# similar words
results = model.wv.most_similar(positive=["hoge"])
for res in results:
    print(res)



