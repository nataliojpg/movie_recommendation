# Movie Recommendations
This is a web-based movie recommendation app built with **Streamlit** and **scikit-learn**, using Netflix's movie data. It allows users to:

- Browse movies one by one
- Mark them as 👍 Liked, 👎 Disliked, or 🤷 Don't Know
- Set their genre preferences
- Receive personalized movie recommendations using machine learning
- 
## How it works
1. **TF-IDF Vectorization**: Movie metadata is converted into numerical vectors.
2. **User Feedback**: You give feedback by liking/disliking/unmarking movies.
3. **Model Training**: A logistic regression model is trained on the labeled data.
4. **Prediction**: The model ranks the remaining movies by how likely you are to like them.
5. **Display**: Top 10 personalized recommendations are shown.
