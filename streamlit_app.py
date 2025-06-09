import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from sentence_transformers import SentenceTransformer, util  # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Nastavenie rozloženia
st.set_page_config(layout="wide")

# Načítanie dát
df = pd.read_csv("data/contract_criteria_final_general_only.csv")
df = df.dropna(subset=["description", "general_criterion"])
df["description"] = df["description"].astype(str).str.strip()

general_criterion_embeddings = torch.tensor(np.load("embeddings/general_criterions_embeddings_split.npy"))
description_embeddings = torch.tensor(np.load("embeddings/descriptions_embeddings_split.npy"))

# Model pre embeddingy
embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device="cpu")

# Mapovanie pre presné priradenie
desc_to_label = {desc.lower(): lab for desc, lab in zip(df["description"], df["general_criterion"])}

# Tréning klasifikačných modelov
X = df["description"]
y = df["general_criterion"]

pipeline_nb = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])
pipeline_lr = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

pipeline_nb.fit(X, y)
pipeline_lr.fit(X, y)

# ====================
# UI Rozhranie Streamlit
# ====================

st.title("Vyhľadávanie a predikcia všeobecných kritérií")

st.header("Vyhľadávanie podobných všeobecných kritérií alebo popisov")
search_column = st.radio("Vyhľadávať podľa:", ["Všeobecné kritérium", "Popis"], horizontal=True)
query = st.text_input("Zadaj text na vyhľadanie alebo predikciu:")
top_k = st.slider("Počet výsledkov", 1, 20, 5)
search_button = st.button("Hľadať podobnosti")

# Funkcia pre podobnosti
def find_similar_unique_items(query, top_k=5, search_column="general_criterion", embeddings=None):
    if embeddings is None:
        return pd.DataFrame()

    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]

    sorted_indices = torch.argsort(cosine_scores, descending=True)
    seen_values = set()
    results = []

    for idx in sorted_indices:
        idx = int(idx)
        general_criterion = df.iloc[idx]["general_criterion"]
        description = df.iloc[idx]["description"]
        label = df.iloc[idx].get("label", "")

        value_to_check = general_criterion if search_column == "general_criterion" else description

        if value_to_check not in seen_values:
            seen_values.add(value_to_check)
            results.append({
                "Všeobecné kritérium": general_criterion,
                "Popis": description,
                "Label": label,
                "Podobnosť": round(cosine_scores[idx].item(), 3)
            })

        if len(results) >= top_k:
            break

    return pd.DataFrame(results)

# Spustenie vyhľadávania
if search_button and query:
    column_key = "description" if search_column == "Popis" else "general_criterion"
    selected_embeddings = description_embeddings if column_key == "description" else general_criterion_embeddings

    with st.spinner("Vyhľadávam podobnosti..."):
        results_df = find_similar_unique_items(query, top_k, search_column=column_key, embeddings=selected_embeddings)

        if not results_df.empty:
            def color_similarity(val):
                if val >= 0.8:
                    return "background-color: #d4edda;"  # zelená
                elif val >= 0.5:
                    return "background-color: #fff3cd;"  # žltá
                else:
                    return "background-color: #f8d7da;"  # červená

            st.success("Výsledky nájdené!")
            styled_df = results_df.style.applymap(color_similarity, subset=["Podobnosť"])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("Nenašli sa žiadne výsledky.")

# ====================
# Sekcia na predikciu
# ====================
st.header("Predikcia všeobecného kritéria")

model_choice = st.radio("Vyber model na predikciu:", ["Logistic Regression", "Naive Bayes"], horizontal=True)
predict_button = st.button("Predikovať")

if predict_button and query:
    key = query.lower().strip()

    if key in desc_to_label:
        st.success(f"Presné priradenie z datasetu: **{desc_to_label[key]}**")
    else:
        model_used = pipeline_lr if model_choice == "Logistic Regression" else pipeline_nb
        pred = model_used.predict([query])[0]
        st.info(f"Predikovaná kategória ({model_choice}): **{pred}**")
