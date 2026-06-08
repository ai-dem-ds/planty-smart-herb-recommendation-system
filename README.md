# 🌺 Planty – Smart Herb Recommendation System

```
https://planty-ai.streamlit.app/
```

Planty is a Streamlit-based herbal wellness recommendation app.  
The app allows users to describe how they feel in natural language and then suggests herbs or plants that may gently support their needs.

This project was created as part of my Data Science & AI final project.


---


## 🌿 Project Idea

Many people are interested in herbs and plant-based wellness, but it can be difficult to understand which plants may be relevant for specific needs.

Planty helps users explore herbal options by connecting user input such as:

- stress
- sleep problems
- digestion issues
- irritated skin
- low energy
- fluid retention
- headache or body discomfort

with a curated herbal dataset.

The goal is not to provide medical advice, but to create a safe, user-friendly recommendation system for herbal wellness orientation.


---


## ⚠️ Important Disclaimer

Planty is for educational and wellness-orientation purposes only.

It does **NOT** replace medical advice, diagnosis, or treatment.  
Users should always review safety information carefully and consult a qualified healthcare professional when needed.


---


## 🚀 Main Features

### 1. Natural Language Input

Users can type short descriptions such as:

- I feel stressed and overwhelmed
- I cannot sleep and feel restless
- My stomach feels heavy and bloated
- My skin feels irritated and dry
- I feel puffy and my fingers are swollen

Planty detects relevant needs from the text and maps them to internal wellness tags.


### 2. Herb Recommendation Cards

The app displays recommendations in clean Streamlit cards.

Each card includes:

- Common plant name
- scientific name
- explanation why the plant was suggested
- possible support area
- usual preparation type
- safety warning level
- recommendation fit
- expandable “Read more” section
- expandable “Safety details” section


### 3. Plant Search

Users can also search for a specific plant by name.

The search result uses the same card layout as the recommendation output and shows:

- plant information
- preparation type
- safety details
- summary information when available


### 4. Recommendation Modes

The app includes different display modes:

- Best matches – shows the strongest matching herbs
- Show all herbs – shows all matching herbs from the dataset and "select the number of recommendation" button loses its functionality
- Surprise me – shows a random selection of matching herbs

This gives the user more control over the recommendation output.


---


## 🧠 Recommendation Logic

### Planty uses a hybrid recommendation approach.

1. Rule-Based Tag Matching

The first layer is based on user need tags.

User input is mapped to tags such as:

- stress
- sleep
- digestion
- skin
- inflammation
- pain
- cramps
- energy
- focus
- immune support
- respiratory support
- edema

These tags are matched with plant tags in the dataset.


### 2. spaCy Text Preprocessing

spaCy is used to improve text understanding.

The user input is normalized with lemmatization, so different word forms can be interpreted more consistently.

Example:

- stressed → stress
- sleeping → sleep
- worried → worry
- swollen → swell

This makes the keyword-based NLP layer more flexible.


### 3. Summary Signal Layer

Planty also analyzes plant summaries and medicinal property text to create additional wellness signals.

Examples of summary signals:

- calming background
- sleep-related background
- digestive background
- skin-related background
- body comfort background

These signals are used to improve the explanation text inside the recommendation cards.


### 4. TF-IDF and Cosine Similarity

A TF-IDF similarity layer was added to compare the user input with plant-related text fields.

The similarity text is built from:

- common name
- medicinal properties
- summary
- user need tags
- effect tags

Cosine similarity is then used to calculate how closely the user input matches each plant description.

The final recommendation score combines:
-> tag-based recommendation score + TF-IDF similarity score

The tag-based logic remains the main ranking system, while TF-IDF supports the ranking slightly.

---


## 🛠️ Technologies Used:

Python
pandas
Streamlit
spaCy
scikit-learn
TF-IDF Vectorizer
Cosine Similarity
Jupyter Notebook
VS Code
Git / GitHub


📁 Project Structure:

```text
smart_herb_recommendation_system/
│
├── app.py
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_dataset_prep.ipynb
│   └── 02_build_reco_function.ipynb
│
└── .streamlit/
    └── config.toml
```

---


## 📊 Dataset

### The project uses a medicinal plant dataset based on plant information such as:

- common name
- scientific name
- medicinal properties
- known hazards
- summary


### The dataset was cleaned and enriched with additional columns for the recommendation system, including:

- user need tags
- effect tags
- preparation type
- warning level
- recommendation reason
- 🔒 Safety Approach

**Safety is an important part of the app.**

### Planty includes:

- warning levels
- safety notes
- known hazards
- safety detail expanders
- soft language to avoid medical claims

**Plants with higher warning levels are handled more carefully in the output.**

The app clearly states that herbal recommendations are for general wellness orientation only.


---


### ▶️ How to Run the App:

1. Clone the repository
    - git clone <your-repository-link>
    - cd smart_herb_recommendation_system
2. Create and activate a virtual environment
    - python3 -m venv .venv
    - source .venv/bin/activate
3. Install dependencies
    - python -m pip install -r requirements.txt
    - The required spaCy English model is included in the requirements file
4. Run the Streamlit app
    - streamlit run app.py


---


## 💬 Example Inputs:

- I feel stressed and overwhelmed
- I cannot sleep and feel restless
- My stomach feels heavy and bloated
- My skin feels irritated and dry
- I feel puffy and my fingers are swollen


---


## ✅ Current Status:

The current version includes:

- working Streamlit app
- plant search
- recommendation cards
- keyword-based NLP
- spaCy text preprocessing
- summary signal layer
- TF-IDF and cosine similarity scoring
- final recommendation ranking
- safety notes and expandable hazard information
- 🔮 Future Improvements


---


## Possible next steps:

- add multilingual support for German and Turkish
- improve semantic matching with sentence transformers and embeddings
- add a chatbot-style conversation layer
- improve UI and mobile responsiveness
- expand the dataset
- add plant image recognition with computer vision
- improve safety filtering and user guidance
- deploy the app online


---


## 👩‍💻 Author
**Kübra Demirhan**<br>
Data Science & AI Final Project<br>
Planty – Smart Herb Recommendation System


---


## 🌺 Final Note
Planty is designed as a gentle herbal wellness recommendation system.
It combines rule-based recommendation logic, NLP preprocessing, plant information, and safety awareness to help users explore herbs in a more structured and understandable way.


