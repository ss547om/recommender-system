# recommender-system
Tool for recommending suitable negotiation criteria as part of a university semester project.

---

### Potrebné balíčky  
- pyreadstat  
- pandas  
- numpy  
- stanza  
- sentence_transformers (SentenceTransformer, util)  
- re  
- streamlit  
- torch
- scikit-learn
- ipywidgets  

---

## Štruktúra projektu

```
data/
    contract_criteria_clean_split.csv
    contract_criteria_export.csv
    criteria_data.csv
    description_data.csv
embeddings/
    criteria_embeddings_split.npy
    descriptions_embeddings_split.npy
    general_criterions_embeddings_split.npy
config.toml
model.ipynb
Príprava_dát1.ipynb
README.md
streamlit_app.py
```

# streamlit_app na vyhľadávanie a predikciu všeobecných kritérií

Táto **Streamlit aplikácia** kombinuje dve hlavné funkcionality:

1. **Sémantické vyhľadávanie podobných všeobecných kritérií alebo popisov** pomocou embeddingov.
2. **Klasifikácia popisu** pomocou ML modelov (Logistic Regression / Naive Bayes) s možnosťou presného priradenia na základe známych vstupov.

---

## Funkcionalita

### Vyhľadávanie podobností

- Výber medzi vyhľadávaním podľa:
  - *Všeobecného kritéria*
  - *Popisu*
- Využitie embeddingov vytvorených pomocou modelu `paraphrase-multilingual-MiniLM-L12-v2`
- Farebne odlíšené výsledky podľa podobnosti:
  - **≥ 0.8** – vysoká podobnosť (zelená)
  - **0.5 – 0.79** – stredná podobnosť (žltá)
  - **< 0.5** – nízka podobnosť (červená)
- Nastaviteľný počet výsledkov (1–20)

### Predikcia kategórie

- Výber modelu: **Logistic Regression** alebo **Naive Bayes**
- Ak sa zadaný popis zhoduje s popisom v dátach, použije sa presné priradenie z datasetu
- Inak sa použije vybraný model na predikciu kategórie

---

## Požiadavky

Inštalácia závislostí:

```bash
pip install streamlit pandas numpy torch scikit-learn sentence-transformers
```

---

# model.ipynb

Tento skript slúži na predspracovanie dát pre aplikácie využívajúce sémantickú analýzu textu. Vygeneruje **embeddingy** pre stĺpce `criterion` a `description` pomocou predtrénovaného modelu `paraphrase-multilingual-MiniLM-L12-v2`.

---

## Funkcionalita

1. **Načíta vstupné dáta** zo stĺpcov `criterion` a `description`.
2. **Vyčistí a prevedie texty na embeddingy** pomocou SentenceTransformer modelu.
3. **Uloží embeddingy** ako `.npy` súbory do priečinka `embeddings/`.
4. **Uloží dáta** s pôvodným textom do CSV súborov v priečinku `data/`.

---

## Požiadavky

```bash
pip install pandas numpy stanza sentence-transformers
```
---
# Príprava_dát1.ipynb

Tento skript slúži na extrakciu a úpravu kritérií verejných obstarávaní zo súboru vo formáte `.sav` (SPSS), ich predspracovanie a uloženie vo forme CSV súborov vhodných pre ďalšie NLP úlohy.

---

## Funkcionalita

1. Načíta `.sav` súbor `contract_eval_CPV_HI.sav`.
2. Vyberie relevantné stĺpce: `contract_id`, `criterion`, `description`, `label`.
3. Uloží ich do CSV: `contract_criteria_export.csv`.
4. Odfiltruje kritériá obsahujúce výraz *"cena"* (a jej varianty).
5. Uloží necenové kritériá do `contract_criteria_necenove.csv`.

---

## Požiadavky

```bash
pip install pandas pyreadstat
```
# Klasifikácia popisov pomocou Naive Bayes a Logistic Regression

# contract_classifier.ipynb

Tento Jupyter notebook slúži na klasifikáciu textových popisov ku všeobecným kritériám verejného obstarávania. Kombinuje tradičné ML modely s možnosťou priameho mapovania známych popisov.

---

## Funkcionalita

- **Načítanie a čistenie dát** zo súboru `contract_criteria_final_general_only.csv`
- **Presné mapovanie** známych popisov – ak sa popis nachádza v dátach, kategória sa priradí priamo
- **Fallback klasifikácia** neznámych popisov pomocou jedného z dvoch modelov:
  - **Naive Bayes**
  - **Logistic Regression** (s `class_weight='balanced'`)
- **Vyhodnotenie** modelov pomocou `classification_report`
- **Interaktívne rozhranie** v Jupyteri pomocou `ipywidgets`

---

## Požiadavky

```bash
pip install pandas scikit-learn ipywidgets
```
