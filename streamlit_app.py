import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("🎬 Movie Recommendations")

@st.cache_data
def get_data():
    df = pd.read_csv("./data/netflix_titles.csv")
    df.drop(df[df['country'].isna()| df['cast'].isna() | df['director'].isna()].index, inplace = True)
    df = df.assign(country=df['country'].str.split(', ')).explode('country')
    df['genre'] = df['listed_in'].str.split(', ')
    df = df.explode('genre')
    df.reset_index(drop=True, inplace=True)
    df.fillna('', inplace=True)
    #df['content'] = df['listed_in'] + ' ' + df['description'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['country']
    df['content'] = df['listed_in'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['country']
    return df

df = get_data()

st.sidebar.header("Preference Filter")

available_genres = sorted(df['genre'].dropna().unique())

# Selección en sidebar
fav_genres = st.sidebar.multiselect(
    "Which genres are you interested in?",
    available_genres,
    default=[]
)

# Filtrar el df por géneros si se seleccionaron
if fav_genres:
    df = df[df['genre'].isin(fav_genres)]


if "likes" not in st.session_state:
    st.session_state.likes = []
if "dislikes" not in st.session_state:
    st.session_state.dislikes = []
if "omitted" not in st.session_state:
    st.session_state.omitted = []


other_movies = df[
    ~df['title'].isin(st.session_state.likes + st.session_state.dislikes + st.session_state.omitted)
].drop_duplicates(subset=['title', 'director'])

if not other_movies.empty:
    peli = other_movies.sample(1).iloc[0]
    st.subheader(peli['title'])
    st.write("🎭 **Genres**:", peli['listed_in'])
    st.write("🎬 **Director**:", peli['director'])
    st.write("👥 **Cast**:", peli['cast'])
    st.write("📝 **Description**:", peli['description'])

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👍 Like it/ Might like"):
            st.session_state.likes.append(peli['title'])
            st.experimental_rerun()
    with col2:
        if st.button("👎 Didn't like it/ Wouldn't like"):
            st.session_state.dislikes.append(peli['title'])
            st.experimental_rerun()
    with col3:
        if st.button("🤷 Don't know"):
            st.session_state.omitted.append(peli['title'])
            st.experimental_rerun()
else:
    st.success("✅ You have give a value to all the movies")


if st.button("🔍 Recommend Movies"):
    df['like'] = df['title'].apply(lambda x: 1 if x in st.session_state.likes else (0 if x in st.session_state.dislikes else None))
    df_train = df[df['like'].notna()]
    if df_train['like'].nunique() < 2:
        st.warning("⚠️ You have to have at least one movie you like and one movie you dislike")
    else:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X_train = vectorizer.fit_transform(df_train['content'])
        y_train = df_train['like']
        
        model = LogisticRegression()
        model.fit(X_train, y_train)

        df_unseen = df[df['like'].isna() & ~df['title'].isin(st.session_state.dislikes)].copy()
        df_unseen = df_unseen.drop_duplicates(subset=['title', 'director'])
        X_unseen = vectorizer.transform(df_unseen['content'])
        df_unseen['like_probability'] = model.predict_proba(X_unseen)[:, 1]

        recomendaciones = df_unseen.sort_values(by='like_probability', ascending=False).head(10)
        st.subheader("🎯 Recommendations")
        for idx, row in recomendaciones.iterrows():
            st.markdown(f"**{row['title']}**") #— _Probability of liking: {row['like_probability']:.2f}_")

# === Mostrar historial ===
if st.checkbox("🗂 My movie history of likes and dislikes"):
    st.write("👍 What I liked:")
    st.write(st.session_state.likes)
    st.write("👎 What I didn't like:")
    st.write(st.session_state.dislikes)
    st.write("🤷 Don't know:")
    st.write(st.session_state.omitted)
