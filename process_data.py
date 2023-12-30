import pandas as pd
import re
import jieba


train_data = pd.read_csv("data/train.csv",delimiter=',', header=0)

#拆分成两部分
texts = train_data.iloc[:,1]
texts = [t.replace(t[2:t.find('，')],"") for t in texts]
for i in range(len(texts)):
    npos = texts[i].find('，')
    name = texts[i][2:npos]
    if len(name) > 1 and len(name) < 5:
        texts[i].replace(name,"")
splits_dicts = ["审理终结", "进行了审理", "进行了审核","审理查明"]
text_pre = []
text_suf = []
for text in texts:
    for spd in splits_dicts:
        pos = text.find(spd)
        if pos > 0:
            text_pre.append(text[:pos])
            text_suf.append(text[pos:])
            break

#罪名提取
crime_features = []

cpa1 = '(以|认定|因|对)(被告.{2,5}|被告人|罪犯.{2,5}|罪犯|被告人因|被告人.{2,5}|该犯|其|被告|上诉人（原审被告人）)?(犯|构成)((.{2,15}?罪?、)*(.{2,15}?罪))'
cpa2 = '(以|认定|因|对)((.{2,15}?罪、)*(.{2,15}?罪))判处'
cpa3 = '(被告.{2,5}|被告人|罪犯.{2,5}|罪犯|被告人因|被告人.{2,5}|该犯|其|被告)(犯|构成)((.{2,15}?罪、)*(.{2,15}?罪))'
pattern = cpa1 +'|'+ cpa2 + '|' + cpa3
 
for id,tp in enumerate(text_pre):
    r = re.search(pattern,tp)
    if r:
        if r.group(4):
            crime_features.append(r.group(4))
        elif r.group(8):
            crime_features.append(r.group(8))
        else:
            crime_features.append(r.group(13))
    else:
        crime_features.append(" ")

#刑期
sentence_features = []
pa = '(判处|执行)(被告.{2,5}|被告人|罪犯.{2,5}|罪犯|被告人因|被告人.{2,5}|该犯|其|被告)?(有期徒刑.{1,6}年|无期徒刑|死刑)'
patterns = pa
for id,tp in enumerate(text_pre):
    r = re.search(pa,tp)
    if r:
        if r.group(3):
            sentence_features.append(r.group(3))
    else:
        sentence_features.append(' ')

#审查部分处理 切词 删词
stopwords = open("停用词表.txt", encoding="utf-8").read().split('\n')
review_cut = []
for ts in text_suf:
    review_cut.append([token for token in jieba.lcut(ts) if token not in stopwords])


#文本特征统一形式
crime_features_cut = [cf.split('、') for cf in crime_features]
sentence_features_cut = [jieba.lcut(sf) for sf in sentence_features]
datas = []
for cf,sf,rev in zip(crime_features_cut,sentence_features_cut,review_cut):
    data = []
    data.extend(cf)
    data.extend(sf)
    data.extend(rev)
    datas.append(data)

#分割存储
datas = [" ".join(d) for d in datas]

df = pd.DataFrame({
    'text':datas,
    'label':train_data.iloc[:,2]
})


data_len = df.shape[0]
train_split = int(data_len * 8 / 11)
dev_split = int(data_len *  10 / 11)
train_datas = df.iloc[:train_split]
dev_datas = df.iloc[train_split:]



train_datas.to_csv("proed_data/train.csv",sep=",", index=False)
dev_datas.to_csv("proed_data/valid.csv",sep=",", index=False)




