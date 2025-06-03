import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from sentence_transformers import SentenceTransformer, util  # type: ignore

# Nastavenie rozloženia
st.set_page_config(layout="wide")

# Načítanie dát
df = pd.read_csv("data/contract_criteria_clean_split.csv")
criteria_embeddings = torch.tensor(np.load("embeddings/criteria_embeddings_split.npy"))
description_embeddings = torch.tensor(np.load("embeddings/descriptions_embeddings_split.npy"))

# Načítanie modelu
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Funkcia na vyhľadanie podobností s unikátnosťou podľa výberu
def find_similar_unique_items(query, top_k=5, search_column="criterion", embeddings=None):
    if embeddings is None:
        return pd.DataFrame()

    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]

    sorted_indices = torch.argsort(cosine_scores, descending=True)
    seen_values = set()
    results = []

    for idx in sorted_indices:
        idx = int(idx)
        criterion = df.iloc[idx]["criterion"]
        description = df.iloc[idx]["description"]
        label = df.iloc[idx].get("label", "")

        # Unikátnosť podľa stĺpca, ktorý používateľ zvolil
        value_to_check = criterion if search_column == "criterion" else description

        if value_to_check not in seen_values:
            seen_values.add(value_to_check)
            results.append({
                "Kritérium": criterion,
                "Popis": description,
                "Label": label,
                "Podobnosť": round(cosine_scores[idx].item(), 3)
            })

        if len(results) >= top_k:
            break

    return pd.DataFrame(results)

# UI
st.title("Vyhľadávanie podobných kritérií alebo popisov")

search_column = st.radio("Vyhľadávať podľa:", ["Kritérium", "Popis"])
query = st.text_input("Zadaj text na vyhľadanie:")
top_k = st.slider("Počet výsledkov", 1, 20, 5)
search_button = st.button("Hľadať")

# Vyhľadanie
if search_button and query:
    column_key = "description" if search_column == "Popis" else "criterion"
    selected_embeddings = description_embeddings if column_key == "description" else criteria_embeddings

    with st.spinner("Vyhľadávam podobnosti..."):
        results_df = find_similar_unique_items(query, top_k, search_column=column_key, embeddings=selected_embeddings)

        if not results_df.empty:
            # Farebné označenie podľa podobnosti
            def color_similarity(val):
                if val >= 0.8:
                    return "background-color: #d4edda;"  # zelená
                elif val >= 0.5:
                    return "background-color: #fff3cd;"  # žltá
                else:
                    return "background-color: #f8d7da;"  # červená

            st.success("Výsledky nájdené!")
            st.markdown("###Tabuľka výsledkov")

            styled_df = results_df.style.applymap(color_similarity, subset=["Podobnosť"])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("Nenašli sa žiadne výsledky.")
