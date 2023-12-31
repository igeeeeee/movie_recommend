import streamlit as st
import pandas as pd
import numpy as np
import gensim

st.title('映画レコメンド')

# 映画情報の読み込み
movies = pd.read_csv("data/movies.tsv", sep="\t")

# 学習済みのitem2vecモデルの読み込み
model = gensim.models.word2vec.Word2Vec.load("data/item2vec.model")

# 映画IDとタイトルを辞書型に変換
movie_titles = movies["title"].tolist()
movie_ids = movies["movie_id"].tolist()
movie_genre =movies["genre"].tolist()
movie_id_to_title = dict(zip(movie_ids, movie_titles))
movie_title_to_id = dict(zip(movie_titles, movie_ids))
movie_id_to_genre = dict(zip(movie_ids, movie_genre))

st.markdown("## 1本の映画に対して似ている映画を表示する")

name = st.text_input('映画の名前を入力(年は不要)', 'Toy Story')
now_title = ""
now_cost = 9999999999
# print(name)
for titl in movie_titles:
    n = len(name)
    tmp_title = ""
    for ch in titl:
        if(ch == '('):
            break
        else:
            tmp_title += ch
    m = len(tmp_title)
    dp =  [[9999999]*(m+2) for _ in range((n+2))]
    dp[0][0] = 0
    #編集距離の計算
    for i in range(n):
        for j in range(m):
            if(name[i] == tmp_title[j]):
                dp[i+1][j+1] = min(dp[i+1][j+1],dp[i][j])
            
            dp[i+1][j+1] = min(dp[i+1][j+1],dp[i+1][j] + 1)
            dp[i+1][j+1] = min(dp[i+1][j+1],dp[i][j+1]+1)
    # print(dp[n][m])
    if(now_cost > dp[n][m]):
        now_title = titl
        now_cost = dp[n][m]

# selected_movie = st.selectbox("映画を選んでください", movie_titles)
selected_movie = now_title
selected_movie_id = movie_title_to_id[selected_movie]

st.write(f"あなたが選択した映画は{selected_movie}(id={selected_movie_id})です")

# 似ている映画を表示
st.markdown(f"### {selected_movie}に似ている映画")
num=int(st.text_input('出力個数を入力', '10'))
results = []
for movie_id, score in model.wv.most_similar(selected_movie_id,topn = num):
    title = movie_id_to_title[movie_id]
    genre = movie_id_to_genre[movie_id]
    genres = ""
    buff = ""
    for ge in genre:
        if(ge == ']' or ge == '[' or ge == ',' or ge == ' '):
            continue
        elif(ge == '\''):
            if(buff == ""):
                continue
            else:
                genres += buff + ', '
                buff = ""
        else:
            buff += ge
    if(len(genres) >= 2):
        genres = genres[:-2]
    results.append({"title": title, "score": score,"genre":genres})
results = pd.DataFrame(results)

st.write(results)

