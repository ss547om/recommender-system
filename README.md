# recommender-system
Tool for recommending suitable negotiation criteria as part of a university semester project.
### Potrebné balíčky  
- pyreadstat  
- pandas  
- numpy  
- stanza  
- sentence_transformers (SentenceTransformer, util)  
- re  
- streamlit  
- torch  

# Aplikácia na vyhľadávanie podobných kritérií a popisov

Táto **Streamlit aplikácia** umožňuje interaktívne vyhľadávať podobné **kritériá** alebo **popisy** pomocou modelu SentenceTransformer a predpočítaných embeddingov.

---

## Funkcionalita

- **Výber typu vyhľadávania** – medzi *Kritérium* a *Popis*
- **Textové vyhľadávanie** – zadanie vlastného dotazu
- **Nastaviteľný počet výsledkov** (1–20)
- **Farebné označenie výsledkov** podľa skóre podobnosti:
  - **≥ 0.8** – vysoká podobnosť
  - **0.5–0.79** – stredná podobnosť
  - **< 0.5** – nízka podobnosť

---

## Použitý model

- **Model:** [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **Embeddingy:** predpočítané `.npy` súbory (kritériá a popisy)
- **Knižnice:** `sentence-transformers`, `torch`, `streamlit`, `pandas`, `numpy`

---

## Štruktúra projektu

´´´
├── app.py # Hlavný skript Streamlit aplikácie  
├── data/  
│ ├── contract_criteria_clean_split.csv # CSV s dátami  
│ ├── criteria_embeddings_split.npy # Embeddingy kritérií  
│ └── descriptions_embeddings_split.npy # Embeddingy popisov  
├── README.md # Dokumentácia  
├── requirements.txt # Zoznam závislostí  
├── .streamlit/  
│ └── config.toml # Nastavenie vzhľadu aplikácie  
´´´
