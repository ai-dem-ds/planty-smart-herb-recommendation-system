import pandas as pd
import streamlit as st
from textwrap import dedent
import random
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load dataset
# -----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/mvp_herbal_dataset_v3.csv")


df = load_data()

#------------------------------

# spaCy NLP
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


# -----------------------------
# Helper functions
# -----------------------------

def split_tags(tag_string):
    if pd.isna(tag_string) or tag_string == "":
        return []
    
    return [tag.strip() for tag in tag_string.split(",")]


def get_matched_tags(user_tags, plant_tags):
    user_tags = set(user_tags)
    plant_tags = set(plant_tags)
    
    matched_tags = user_tags.intersection(plant_tags)
    
    return list(matched_tags)


def generate_recommendation_reason(matched_tags):
    if len(matched_tags) == 0:
        return "No strong match found."
    
    tag_description = {
        "digestion": "digestive support",
        "pain": "pain-related support",
        "cramps": "cramp-related support",
        "menstrual_support": "menstrual discomfort support",
        "stress": "stress support",
        "sleep": "sleep support",
        "skin": "skin-related support",
        "inflammation": "inflammation-related support",
        "anxiety": "anxiety or nervousness support",
        "energy": "energy support",
        "relaxation": "relaxation support"
    }
    
    readable = [
        tag_description.get(tag, tag.replace("_", " "))
        for tag in matched_tags
    ]

    readable_text = ", ".join(readable)

    return f"This plant may be helpful for {readable_text}."


def generate_safety_note(warning_level):
    if warning_level == "low":
        return "Low warning level based on available hazard information."
    elif warning_level == "medium":
        return "Medium warning level. Please review known hazards before use."
    elif warning_level == "high":
        return "High warning level. This plant is not recommended."
    else:
        return "Warning level unknown."
    

def format_score(score):
    if score == 1.0:
        return "Very strong match"
    elif score >= 0.6:
        return "Good match"
    elif score > 0:
        return "Partial match"
    else:
        return "No match"
    

def get_warning_badge(warning_level):
    if warning_level == "low":
        return "🟢 Low warning level"
    elif warning_level == "medium":
        return "🟡 Medium warning level"
    elif warning_level == "high":
        return "🔴 High warning level"
    else:
        return "⚪ Unknown warning level"
    

def format_support_areas(tags):
    if isinstance(tags, list):
        tag_list = tags
    elif isinstance(tags, str):
        tag_list = tags.split(", ")
    else:
        return "Not specified ⚡️"
    
    tag_labels = {
        "stress": "Stress support",
        "sleep": "Sleep support",
        "digestion": "Digestion support",
        "pain": "Pain support",
        "skin": "Skin support",
        "inflammation": "Inflammation support",
        "anxiety": "Calmness support",
        "relaxation": "Relaxation support",
        "energy": "Energy support",
        "focus": "Focus support",
        "immune_support": "Immune support",
        "respiratory_support": "Respiratory support",
        "edema": "Fluid balance support",
        "cramps": "Cramp support",
        "menstural_support": "Menstrual comfort support",
        "mood": "Mood support",
        "headache": "Head comfort support"
    }

    readable_tags = []

    for tag in tag_list:
        clean_tag = str(tag).strip()
        readable_tags.append(
            tag_labels.get(clean_tag, clean_tag.replace("_", " ").title())
        )
    
    return ", ".join(readable_tags)


#---------------------------------------------------------------------------------


# Preparation Format output   
def format_preparation_type(preparation_type):
    if pd.isna(preparation_type):
        return "Not specified ⚡️"
    
    preparation_type = str(preparation_type).strip()

    if preparation_type == " " or preparation_type.lower() == "unknown":
        return "Not specified"
    
    if "not recommended" in preparation_type.lower():
        return "Not recommended due to safety concerns"
    
    preparation_labels = {
        "tea": "Tea",
        "oil": "Oil",
        "topical": "Topical use",
        "food": "Food-related use",
        "extract": "Extract",
        "poultice": "Poultice/Compress",
        "capsule_or_powder": "Capsule or Powder",
        "cream_or_onitment": "Cream or Onitment",
        "juice_or_gel": "Juice or Gel"
    }

    forms = [form.strip() for form in preparation_type.split(",")]
    readable_forms = []

    for form in forms:
        readable_forms.append(
            preparation_labels.get(form, form.replace("_", " ").title())
        )

    return ", ".join(readable_forms)


#---------------------------------------------------------------------------------


# Plant Properties Format
def format_plant_properties(properties):
    if pd.isna(properties) or str(properties).strip() == "":
        return "Info soon available ⏰"
    
    properties_list = [
        prop.strip()
        for prop in str(properties).replace(", ", ";").split(";")
        if prop.strip() != ""
    ]
    
    hidden_terms = ["Cancer"]

    properties_list = [
        prop for prop in properties_list
        if prop not in hidden_terms
    ]

    if len(properties_list) == 0:
        return "Info not available yet ⏰"
    
    return ", ".join(properties_list[:6])


#---------------------------------------------------------------------------------


# Generating Soft Plant Note
def generate_soft_plant_note(row):
    plant_name = row["Common Name"]
    properties = row.get("Medicinal Properties", "")

    if pd.isna(properties) or str(properties).strip() == "":
        return (
            f"Planty found {plant_name}. More detailed wellness information "
            "can be added later."
        )

    text = str(properties).lower()

    support_notes = []

    if "stomachic" in text or "carminative" in text or "digestive" in text:
        support_notes.append("digestive comfort")

    if "sedative" in text or "nervine" in text or "antianxiety" in text:
        support_notes.append("calmness and relaxation")

    if "aromatic" in text or "aromatherapy" in text:
        support_notes.append("aromatic wellness use")

    if "antiinflammatory" in text or "antirheumatic" in text:
        support_notes.append("body comfort")

    if "antibacterial" in text or "antiseptic" in text:
        support_notes.append("traditional cleansing support")

    if len(support_notes) == 0:
        return (
            f"{plant_name} has several traditional herbal properties. "
            "Planty can show more detailed explanations as the app improves."
        )

    support_text = ", ".join(support_notes)

    return (
        f"{plant_name} is associated with {support_text}. "
        "This information is for gentle orientation only, NOT medical advice."
    )


#---------------------------------------------------------------------------------


# Soft Prep Type Format
def format_preparation_type_soft(preparation_type):
    formatted = format_preparation_type(preparation_type)

    if formatted == "Not specified":
        return "No clear preparation form is available yet."

    if "Not recommended" in formatted:
        return "Not suggested for use in this app because the available safety information indicates higher caution."

    return formatted


#---------------------------------------------------------------------------------


# Extract Summary Signals
def extract_summary_signals(row):
    summary = row.get("Summary", "")
    properties = row.get("Medicinal Properties", "")

    summary_text = str(summary).lower() if pd.notna(summary) else ""
    property_text = str(properties).lower() if pd.notna(properties) else ""

    combined_text = summary_text + " " + property_text

    signals = []

    # Calmness / relaxation
    if any(word in combined_text for word in [
        "calm", "calming", "relax", "relaxing", "nervous",
        "nervine", "sedative", "anxiety", "stress", "soothing"
    ]):
        signals.append("calming background")

    # Sleep
    if any(word in combined_text for word in [
        "sleep", "insomnia", "rest", "restful", "night",
        "sedative", "hypnotic"
    ]):
        signals.append("sleep-related background")

    # Digestion
    if any(word in combined_text for word in [
        "digest", "digestion", "digestive", "stomach", "bloating",
        "gas", "carminative", "stomachic", "nausea", "antiemetic"
    ]):
        signals.append("digestive background")

    # Skin / topical comfort
    if any(word in combined_text for word in [
        "skin", "wound", "burn", "irritation", "eczema",
        "emollient", "vulnerary", "antiseptic", "topical"
    ]):
        signals.append("skin-related background")

    # Pain / cramps / body comfort
    if any(word in combined_text for word in [
        "pain", "ache", "cramp", "cramps", "antispasmodic",
        "analgesic", "anodyne", "antirheumatic", "inflammation",
        "antiinflammatory"
    ]):
        signals.append("body comfort background")

    # Respiratory / seasonal wellness
    if any(word in combined_text for word in [
        "cough", "respiratory", "breathing", "bronchitis",
        "expectorant", "cold", "flu", "antiviral"
    ]):
        signals.append("respiratory or seasonal wellness background")

    # Energy / tonic
    if any(word in combined_text for word in [
        "tonic", "stimulant", "fatigue", "energy", "vitality"
    ]):
        signals.append("vitality background")

    # Remove duplicates while keeping order
    signals = list(dict.fromkeys(signals))

    return signals


#---------------------------------------------------------------------------------

# Soft Safety Note
def generate_safety_note_soft(warning_level):
    if warning_level == "low":
        return "This plant currently has a low warning level. Still, herbal products should be used mindfully."
    elif warning_level == "medium":
        return "Please use this plant with care and review the available safety information before use."
    elif warning_level == "high":
        return "Planty does not recommend this plant because the available safety information suggests a higer risk."
    else:
        return "Safety information is limited. Please use caution and review the available details."
    

#---------------------------------------------------------------------------------



# Generating Soft Recommendation Note for Output from Chatbox
# old version - currently not used
def generate_soft_recommendation_note(row):
    plant_name = row["Common Name"]

    matched_tags = row.get("matched_tags", [])

    if isinstance(matched_tags, str):
        matched_tags = [tag.strip() for tag in matched_tags.split(",") if tag.strip() != ""]

    readable_tags = format_support_areas(matched_tags)

    properties = row.get("Medicinal Properties", "")
    summary = row.get("Summary", "")

    if pd.notna(summary) and str(summary).strip() != "":
        summary_text = str(summary).lower()
    else:
        summary_text = ""

    property_text = str(properties).lower()

    soft_reasons = []

    if "stress" in matched_tags or "relaxation" in matched_tags or "anxiety" in matched_tags:
        if "sedative" in property_text or "nervine" in property_text or "relax" in summary_text:
            soft_reasons.append("calmness and relaxation")
        else:
            soft_reasons.append("emotional balance")

    if "sleep" in matched_tags:
        if "sedative" in property_text or "nervine" in property_text or "sleep" in summary_text:
            soft_reasons.append("restful sleep support")
        else:
            soft_reasons.append("evening relaxation")

    if "digestion" in matched_tags:
        if "stomachic" in property_text or "carminative" in property_text or "digest" in summary_text:
            soft_reasons.append("digestive comfort")
        else:
            soft_reasons.append("digestive support")

    if "pain" in matched_tags or "cramps" in matched_tags:
        if "antispasmodic" in property_text:
            soft_reasons.append("cramp-related comfort")
        elif "anodyne" in property_text or "analgesic" in property_text:
            soft_reasons.append("pain-related comfort")
        else:
            soft_reasons.append("body comfort")

    if "skin" in matched_tags or "inflammation" in matched_tags:
        if "antiinflammatory" in property_text or "antiseptic" in property_text:
            soft_reasons.append("skin and body comfort")
        else:
            soft_reasons.append("gentle skin-related support")

    if len(soft_reasons) == 0:
        return (
            f"Planty suggests {plant_name} because it matches your selected support areas: "
            f"{readable_tags}."
        )

    reason_text = ", ".join(list(dict.fromkeys(soft_reasons)))

    return (
        f"Planty suggests {plant_name} because it may support {reason_text}. "
        "This is gentle orientation only and does NOT replace medical advice."
    )

#---------------------------------------------------------------------------------


# Generating Wellness Reason
def generate_wellness_reason(row):
    plant_name = row.get("Common Name", "This plant")

    matched_tags = row.get("matched_tags", [])
    properties = row.get("Medicinal Properties", "")
    summary = row.get("Summary", "")

    summary_signals = extract_summary_signals(row)

    # Make sure matched_tags is always a list
    if isinstance(matched_tags, str):
        matched_tags = [
            tag.strip()
            for tag in matched_tags.split(",")
            if tag.strip() != ""
        ]

    property_text = str(properties).lower() if pd.notna(properties) else ""
    summary_text = str(summary).lower() if pd.notna(summary) else ""

    wellness_notes = []

    # Stress / anxiety / relaxation
    if any(tag in matched_tags for tag in ["stress", "anxiety", "relaxation", "nervousness"]):
        if any(word in property_text for word in ["sedative", "nervine", "antianxiety", "calmative"]):
            wellness_notes.append("calmness, emotional balance and gentle relaxation")
        elif any(word in summary_text for word in ["stress", "anxiety", "relax", "nervous"]):
            wellness_notes.append("a calmer and more grounded feeling")
        else:
            wellness_notes.append("emotional balance and gentle relaxation")

    # Sleep
    if "sleep" in matched_tags:
        if any(word in property_text for word in ["sedative", "nervine", "hypnotic"]):
            wellness_notes.append("evening calm and sleep-related support")
        elif any(word in summary_text for word in ["sleep", "rest", "relax", "night"]):
            wellness_notes.append("a more restful evening routine")
        else:
            wellness_notes.append("restful sleep support")

    # Digestion
    if "digestion" in matched_tags:
        if any(word in property_text for word in ["stomachic", "carminative", "digestive", "antiemetic"]):
            wellness_notes.append("digestive comfort and stomach balance")
        elif any(word in summary_text for word in ["digestion", "stomach", "bloating", "digestive"]):
            wellness_notes.append("digestive comfort")
        else:
            wellness_notes.append("digestive support")

    # Pain / cramps
    if any(tag in matched_tags for tag in ["pain", "cramps", "menstrual_support"]):
        if "antispasmodic" in property_text:
            wellness_notes.append("cramp-related body comfort")
        elif any(word in property_text for word in ["analgesic", "anodyne", "antirheumatic"]):
            wellness_notes.append("pain-related body comfort")
        else:
            wellness_notes.append("general body comfort")

    # Skin / inflammation
    if any(tag in matched_tags for tag in ["skin", "inflammation"]):
        if any(word in property_text for word in ["antiinflammatory", "antiseptic", "vulnerary", "emollient"]):
            wellness_notes.append("gentle skin and body comfort")
        elif any(word in summary_text for word in ["skin", "wound", "irritation", "inflammation"]):
            wellness_notes.append("skin-related comfort")
        else:
            wellness_notes.append("skin-related support")

    # Energy / focus
    if any(tag in matched_tags for tag in ["energy", "focus"]):
        if any(word in property_text for word in ["tonic", "stimulant"]):
            wellness_notes.append("vitality, focus and natural energy")
        else:
            wellness_notes.append("mental clarity and everyday energy")

    # Respiratory / immune
    if any(tag in matched_tags for tag in ["respiratory_support", "immune_support"]):
        if any(word in property_text for word in ["expectorant", "antibacterial", "antiseptic", "antiviral"]):
            wellness_notes.append("traditional respiratory or immune-related support")
        else:
            wellness_notes.append("seasonal wellness support")

    # Fallback if no detailed wellness notes were found
    if len(wellness_notes) == 0:
        support_text = format_support_areas(matched_tags)

        return (
            f"Planty suggests {plant_name} because it gently connects with "
            f"{support_text}. This suggestion is meant as herbal orientation "
            "and does NOT replace medical advice."
        )

    
    # Remove duplicates while keeping order
    wellness_notes = list(dict.fromkeys(wellness_notes))
    wellness_text = ", ".join(wellness_notes)

    # If no detailed wellness notes were created, use matched tags as fallback
    if wellness_text.strip() == "":
        wellness_text = format_support_areas(matched_tags)

    # If even that is empty, use a final safe fallback
    if wellness_text.strip() == "" or wellness_text == "Not specified ⚡":
        wellness_text = "your selected support area"

    # If Summary exists, mention Read More
    if summary_text.strip() != "" and len(summary_text) > 40:
        if len(summary_signals) > 0:
            signal_text = ", ".join(summary_signals[:2])

            return (
                f"Planty suggests {plant_name} because it is traditionally connected with "
                f"{wellness_text}. The plant information also points toward {signal_text}. "
                "You can open 'Read more' below if you would like a deeper look at this plant. "
                "This is wellness guidance only and does NOT replace medical advice."
            )

        return (
            f"Planty suggests {plant_name} because it is traditionally connected with "
            f"{wellness_text}. You can open 'Read more' below if you would like a deeper look "
            "at this plant. This is wellness guidance only and does NOT replace medical advice."
        )

    # Final fallback when there is no Summary
    return (
        f"Planty suggests {plant_name} because it is traditionally connected with "
        f"{wellness_text}. This is wellness guidance only and does NOT replace "
        "medical advice."
    )




#---------------------------------------------------------------------------------


# Search field for specific Plant Search
def search_plant_by_name(dataset, search_query):
    if search_query.strip() == "":
        return pd.DataFrame()
    
    search_query = search_query.lower().strip()

    results = dataset[
        dataset["Common Name"].str.lower().str.contains(search_query, na=False) |
        dataset["Scientific Name"].str.lower().str.contains(search_query, na=False)
    ]

    return results




# Simple keyword-based NLP 

user_need_keywords = {
    "digestion": [
        "stomach", "belly", "digest", "digestion", "nausea", "bloating", "gas", "cramps", "constipation", "diarrhea", "stomach pain", "upset stomach", "indigestion", "heartburn", "acid reflux", "reflux", "burping", "belching", "fullness", "heavy stomach", "stomach disconfort", "acid", "gassy", "bloated"
    ],
    "pain": [
        "pain", "ache", "headache", "cramps", "sore", "hurt"
    ],
     "cramps": [
        "cramp", "cramps", "period cramps", "menstrual cramps", "period pain", "mensturation pain", "menstrual pain"
    ],
    "menstrual support": [
        "period", "menstruation", "menstrual", "pms", "irregular period", "period cramps", "cycle", "menstrual cramps", "hormonal", "period pain", "spotting"
    ],
    "stress": [
        "stress", "stressed", "overwhelmed", "pressure", "tense", "burnout", "mental pressure", "too much", "emotionally tired", "under pressure", "overloaded", "mental overloaded", "worked up", "can't relax", "cannot relax", "restless mind"
    ],
    "sleep": [
        "sleep", "insomnia", "can't sleep", "cannot sleep", "sleepless", "tired at night", "restless", "restless night", "wake up", "waking up", "bad sleep", "trouble sleeping", "difficulty sleeping", "light sleep", "poor sleep", "can't fall asleep", "cannot fall asleep", "wake up often during night"
    ],
    "anxiety": [
        "anxiety", "anxious", "worry", "worried", "nervous", "panic", "inner tension", "overthinking", "uneasy", "on edge", "racing thoughts", "fearful", "restless inside"
    ],
    "relaxation": [
        "relax", "relaxation", "calm", "calming", "restless", "unwind", "slow down", "need calm", "need rest", "peaceful"
    ],
    "skin": [
        "skin", "rash", "wound", "burn", "eczema", "itchy", "irritation", "dry skin", "red skin", "inflamed skin", "acne", "pimple", "pimples", "breakout", "blemish", "oily skin", "sensitive skin", "flaky skin", "peeling skin", "cracked skin", "spots", "skin redness", "skin feel dry", "skin feel irritated"
    ],
    "inflammation": [
        "inflammation", "inflamed", "swelling", "anti inflammatory", "painful swelling", "redness", "heat", "warm skin", "tender"
    ],
    "immune_support": [
        "immune", "immunity", "cold", "flu", "infection"
    ],
    "respiratory_support": [
        "cough", "breathing", "respiratory", "asthma", "bronchitis"
    ],
    "energy": [
        "energy", "tired", "fatigue", "weak", "exhausted", "low energy", "no energy", "drained", "sluggish", "can't get going", "worn out", "run down"
    ],
    "focus": [
        "focus", "concentration", "memory", "mental clarity", "brain fog", "cannot focus", "forgetful", "forgetfulness", "distracted", "mental fatique", "hard to think", "can't think clearly", "mind feels cloudy", "confused"
    ],
    "edema": [
        "edema", "water retention", "swelling", "fluid retention", "puffy", "bloated face", "swollen ankles", "swollen feet", "puffiness", "holding water", "heavy legs", "swollen tights", "swollen hands", "puffy hands", "swollen fingers", "puffy fingers", "heavy hands", "hand swelling", "finger swelling", "ring feels tight", "tight rings", "my skin feels tight", "my fingers feels tight", "my hand feel swollen", "my rings don't fit", "can't get my rings off", "feeling puffy", "feeling swollen"
    ],
    "mood": [
        "sad", "down", "unhappy", "low mood", "emotional", "crying", "tearful", "irritable", "grumpy", "mood swings", "sensitive", "mentally low"
    ],
    "headache": [
        "headache", "migraine", "migraine attack", "head pain", "pressure in head", "tension headache", "heavy head", "pounding head", "head pressure"
    ],

}


# SpaCy Function
def normalize_user_text_with_spacy(user_text):
    if pd.isna(user_text) or str(user_text).strip() == "":
        return ""
    
    user_text = str(user_text).lower().strip()

    user_text = user_text.replace("can't", "cannot")
    user_text = user_text.replace("cant", "cannot")
    user_text = user_text.replace("n't", "not")

    doc = nlp(user_text)

    useful_tokens = []

    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue

        lemma = token.lemma_.lower().strip()

        if lemma != "":
            useful_tokens.append(token.lemma_.lower())
    
    normalized_text = " ".join(useful_tokens)

    return normalized_text



# Version 2 keyword-based NLP function - New version
def extract_tags_from_user_text(user_text):
    if pd.isna(user_text) or str(user_text).strip() == "":
        return []
    
    user_text = str(user_text).lower().strip()
    
    normalized_text = normalize_user_text_with_spacy(user_text)

    search_text = user_text + " " + normalized_text

    extracted_tags = []

    for tag, keywords in user_need_keywords.items():
        for keyword in keywords:
            keyword = keyword.lower().strip()

            if keyword in search_text:
                extracted_tags.append(tag)
                break
    # Map soft user words to existing recommendation tags
    if "headache" in extracted_tags:
        if "pain" not in extracted_tags:
            extracted_tags.append("pain")
        extracted_tags.remove("headache")

    if "mood" in extracted_tags:
        if "stress" not in extracted_tags:
            extracted_tags.append("stress")
        if "relaxation" not in extracted_tags:
            extracted_tags.append("relaxation")
        extracted_tags.remove("mood")
    
    return sorted(set(extracted_tags))


#---------------------------------------------------------------------------------


# Building plant similarity
def build_similarity_text(row):
    parts = []

    for col in ["Common Name", "Medicinal Properties", "Summary", "user_need_tags", "effect_tags"]:
        if col in row.index:
            value = row.get(col, "")
            if pd.notna(value) and str(value).strip() != "":
                parts.append(str(value))

    return " ".join(parts).lower()



# TF-IDF Similarity Function
def add_similarity_scores(user_text, recommendations, detected_tags=None):
    if recommendations is None or recommendations.empty:
        return recommendations

    recommendations = recommendations.copy()

    if pd.isna(user_text) or str(user_text).strip() == "":
        recommendations["similarity_score"] = 0.0
        recommendations["similarity_score_normalized"] = 0.0
        return recommendations

    recommendations["similarity_text"] = recommendations.apply(
        build_similarity_text,
        axis=1
    )

    user_text_clean = str(user_text).lower().strip()

    # Add detected tags to the user text
    if detected_tags is not None and len(detected_tags) > 0:
        detected_tag_text = " ".join(detected_tags)
        user_text_clean = user_text_clean + " " + detected_tag_text

    documents = [user_text_clean] + recommendations["similarity_text"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    user_vector = tfidf_matrix[0:1]
    plant_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(user_vector, plant_vectors).flatten()

    # First create raw similarity score
    recommendations["similarity_score"] = similarities

    # Then normalize similarity score
    max_similarity = recommendations["similarity_score"].max()

    if max_similarity > 0:
        recommendations["similarity_score_normalized"] = (
            recommendations["similarity_score"] / max_similarity
        )
    else:
        recommendations["similarity_score_normalized"] = 0.0

    # Combined final score
    # Main logic stays tag-based, TF-IDF only supports the ranking slightly
    recommendations["final_score"] = (
        recommendations["recommendation_score"] * 0.85
        + recommendations["similarity_score_normalized"] * 0.15
    )

    return recommendations




# -----------------------------
# Recommendation function
# -----------------------------


# Refined Recommendation Function - v2
def recommend_plants_v2(user_tags, dataset, top_n=None):
    recommendations = dataset.copy()

    recommendations["plant_tag_list"] = recommendations["user_need_tags"].apply(split_tags)

    recommendations["matched_tags"] = recommendations["plant_tag_list"].apply(
        lambda plant_tags: get_matched_tags(user_tags, plant_tags)
    )

    recommendations["recommendation_score"] = recommendations["matched_tags"].apply(
        lambda matched_tags: len(matched_tags) / len(user_tags) if len(user_tags) > 0 else 0
    )

    recommendations = recommendations[
        recommendations["warning_level"] != "high"
    ]

    recommendations = recommendations[
        recommendations["recommendation_score"] > 0
    ]

    recommendations["recommendation_reason"] = recommendations["matched_tags"].apply(
        generate_recommendation_reason
    )

    recommendations["safety_note"] = recommendations["warning_level"].apply(
        generate_safety_note
    )

    recommendations = recommendations.sort_values(
        by="recommendation_score",
        ascending=False
    )

    output_columns = [
        "Common Name",
        "Scientific Name",
        "matched_tags",
        "user_need_tags",
        "effect_tags",
        "preparation_type",
        "warning_level",
        "recommendation_score",
        "recommendation_reason",
        "safety_note",
    ]

    if "Summary" in recommendations.columns:
        output_columns.append("Summary")

    if "Known Hazards" in recommendations.columns:
        output_columns.append("Known Hazards")

    if "Medicinal Properties" in recommendations.columns:
        output_columns.append("Medicinal Properties")

    recommendations = recommendations[output_columns]

    if top_n is not None:
        recommendations = recommendations.head(top_n)

    return recommendations


#---------------------------------------------------------------------------------


def recommend_from_user_text(user_text, dataset, top_n=None):
    detected_tags = extract_tags_from_user_text(user_text)

    if len(detected_tags) == 0:
        return detected_tags, None

    recommendations = recommend_plants_v2(
        user_tags=detected_tags,
        dataset=dataset,
        top_n=top_n
    )

    return detected_tags, recommendations



# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Planty",
    page_icon="🌺",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------


st.markdown("""
<style>
/* Allgemeine Button-Form */
div.stButton > button {
    border-radius: 14px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    border: 1px solid transparent;
    transition: all 0.2s ease-in-out;
}

/* Beispiel-Buttons: 4 verschiedene Farben */
div[data-testid="column"]:nth-of-type(1) div.stButton > button {
    background-color: #FADADD;
    color: #7A3E48;
    border: 1px solid #E8B7C0;
}
div[data-testid="column"]:nth-of-type(1) div.stButton > button:hover {
    background-color: #F6C6D0;
    color: #6A2F39;
}

div[data-testid="column"]:nth-of-type(2) div.stButton > button {
    background-color: #E6E0F8;
    color: #5C4B8A;
    border: 1px solid #D1C7F2;
}
div[data-testid="column"]:nth-of-type(2) div.stButton > button:hover {
    background-color: #D9D0F4;
    color: #4E3E7C;
}

div[data-testid="column"]:nth-of-type(3) div.stButton > button {
    background-color: #FBE9D7;
    color: #8A5A2E;
    border: 1px solid #F1D2B2;
}
div[data-testid="column"]:nth-of-type(3) div.stButton > button:hover {
    background-color: #F7DFC7;
    color: #74471F;
}

div[data-testid="column"]:nth-of-type(4) div.stButton > button {
    background-color: #DDF2E1;
    color: #3E6B4A;
    border: 1px solid #BFE3C8;
}
div[data-testid="column"]:nth-of-type(4) div.stButton > button:hover {
    background-color: #CFEAD6;
    color: #2F593A;
}

/* Submit-Button */
div[data-testid="stFormSubmitButton"] > button {
    background-color: #DCECD4;
    color: #2E5032;
    border: 1px solid #B8D3AE;
    border-radius: 14px;
    font-weight: 700;
    padding: 0.65rem 1.2rem;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #CDE3C1;
    color: #234027;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    # st.title("🌺 Planty")
    # # st.write("### Your herbal wellness companion")

    # st.markdown("---")

    st.subheader("🌿 What Planty does")
    st.write(
        "Share how you feel, and Planty suggests herbs that may support your wellbeing ✨"
    )

    st.info(
        "Herbal Guidance Only. It does NOT replace medical advice!"
    )
    st.markdown("---")

    st.subheader("Search Plant 🔍")
    plant_query = st.text_input(
        "Type in the Plant name 🍀",
        placeholder="plant name here",
        key="sidebar_plant_search_query"
    )

    st.markdown("---")

    st.subheader("✨ How it works ✨")
    st.write("1. Describe your symptoms 💭")
    st.write("2. Planty finds matching herbs 🌱")
    st.write("3. You get suitable plant suggestions 🧾")

    st.markdown("---")

    st.subheader("💌 Contact:")

    st.markdown(
        """
        **Email:** planty@e-mail.com
        **Owner:** Kübra Demirhan
        """
    )


# -----------------------------
# Main Page
# -----------------------------

st.title("🌺 Planty")

st.markdown("### Find herb suggestions based on how you feel 🌱")

st.write(
    "Describe your symptoms or needs and Planty will suggest herbs and plants that may support your needs."
)


if plant_query.strip() != "":
    plant_results = search_plant_by_name(df, plant_query)

    if plant_results.empty:
        st.warning("Plant not found 😔")
    else:
        st.success(f"Found {len(plant_results)} ✨")

        for _, row in plant_results.head(5).iterrows():
            with st.container(border=True):
                st.markdown(f"### 🌱 {row['Common Name']}")
                st.caption(f"Scientific name: {row['Scientific Name']}")

                # st.markdown("#### What Planty found")
                # st.write(format_plant_properties(row["Medicinal Properties"]))
                if "matched_tags" not in row.index:
                    row["matched_tags"] = split_tags(row.get("user_need_tags", ""))

                st.markdown("#### How it may support you")
                st.write(generate_wellness_reason(row))

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**How it is usually used**")
                    st.write(format_preparation_type_soft(row["preparation_type"]))

                with col2:
                    st.markdown("**Gentle safety note**")
                    st.write(get_warning_badge(row["warning_level"]))
                    st.write(generate_safety_note_soft(row["warning_level"]))

                summary_text = row.get("Summary", "")

                if isinstance(summary_text, pd.Series):
                    summary_text = summary_text.iloc[0]

                if pd.notna(summary_text) and str(summary_text).strip() != "":
                    with st.expander("Read more"):
                        st.write(summary_text)

                hazard_text = row.get("Known Hazards", "")

                if isinstance(hazard_text, pd.Series):
                    hazard_text = hazard_text.iloc[0]

                if (
                    pd.notna(hazard_text)
                    and str(hazard_text).strip() != ""
                    and str(hazard_text).lower() != "none known"
                ):
                    with st.expander("Safety details"):
                        st.write("Please review this safety information carefully:")
                        st.write(hazard_text)
                    
st.markdown("---")



# -----------------------------
# Input Area
# -----------------------------


# new
st.markdown("#### For a quick start 💭 ")

if "user_input_text" not in st.session_state:
    st.session_state.user_input_text = ""

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Stress 😣"):
        st.session_state.user_input_text = "I feel stressed and overwhelmed"

with col2:
    if st.button("Sleep 🫩"):
        st.session_state.user_input_text = "I cannot sleep and feel restless"

with col3:
    if st.button("Digestion ⚡️"):
        st.session_state.user_input_text = "I have stomach pain and bloating"

with col4:
    if st.button("Skin 🧴"):
        st.session_state.user_input_text = "My skin is irritated and inflamed"

st.markdown("---")


if "surprise_batch_counter" not in st.session_state:
    st.session_state.surprise_batch_counter = 0

if "last_view_mode" not in st.session_state:
    st.session_state.last_view_mode = ""



# old 
with st.form("recommendation_form"):
    user_text = st.text_input(
        "Tell Planty how you're feeling today 🌸",
        placeholder="Chat with Planty 🌱",
        key="user_input_text"
    )

    view_mode = st.selectbox(
        "Select:",
        options=[
            "Best matches ✅",
            "Show all Herbs 📚",
            "Surprise me 🪄"
        ],
        index=0
    )

    if view_mode == "Show all herbs 📚":
        top_n = None
        st.caption("Planty will show you all matching herbs 🌱")
    else:
        top_n = st.selectbox(
        "Select the number of recommendations:",
        options=[2, 3, 5, 7, 10],
        index=1
    )

    submitted = st.form_submit_button("Find Herbs ✨")
    if submitted and view_mode == "Surprise me 🪄":
        st.session_state.surprise_batch_counter += 1


# -----------------------------
# Recommendation Output
# -----------------------------


if submitted:

    if user_text.strip() == "":
        st.error("Please enter a short description.")

    else:
        detected_tags, recommendations = recommend_from_user_text(
            user_text=user_text,
            dataset=df,
            top_n=None
        )

        recommendations = add_similarity_scores(
            user_text=user_text,
            recommendations=recommendations,
            detected_tags=detected_tags
        )

        recommendations = recommendations.sort_values(
            by="final_score",
            ascending=False
        )

        if len(detected_tags) == 0:
            st.error(
                "No needs detected. Please try words like sleep, stress, pain, skin, inflammation, digestion or anxiety."
            )

        elif recommendations is None or recommendations.empty:
            st.warning("No suitable recommendations found after safety filtering.")

        else:
            st.success(f"Planty noticed: {', '.join(detected_tags)}")

            # Make sure top_n is an integer
            selected_number = int(top_n)

            # Create a separate display dataframe
            recommendations_to_show = recommendations.copy()

            # Mode 1: Best matches
            if view_mode == "Best matches ✅":
                recommendations_to_show = recommendations_to_show.head(selected_number)

            # Mode 2: Show all herbs
            elif view_mode == "Show all herbs 📚":
                recommendations_to_show = recommendations_to_show

            # Mode 3: Surprise me
            elif view_mode == "Surprise me 🪄":
                recommendations_to_show = recommendations_to_show.sample(
                    n=min(selected_number, len(recommendations_to_show)),
                    random_state=random.randint(1, 1_000_000)
                )

            for _, row in recommendations_to_show.iterrows():
                warning_badge = get_warning_badge(row["warning_level"])
                match_quality = format_score(row["recommendation_score"])

                with st.container(border=True):
                    st.markdown(f"### 🌱 {row['Common Name']}")
                    st.caption(f"Scientific name: {row['Scientific Name']}")
                    # please remove text similarity + final score line after tested!
                    # st.caption(f"Text similarity: {row.get('similarity_score', 0):.4f}")
                    # st.caption(f"Final score: {row.get('final_score', 0):.2f}")

                    st.markdown("### Why Planty suggests this 💡")
                    st.write(generate_wellness_reason(row))

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**What it may support:**")
                        st.write(format_support_areas(row["matched_tags"]))

                        st.markdown("**How it is usually used:**")
                        st.write(format_preparation_type_soft(row["preparation_type"]))

                    with col2:
                        st.markdown("**Gentle safety note:**")
                        st.write(get_warning_badge(row["warning_level"]))
                        st.write(generate_safety_note_soft(row["warning_level"]))

                        st.markdown("**Recommendation fit**")
                        st.progress(float(row["recommendation_score"]))
                        st.caption(match_quality)

                    summary_text = row.get("Summary", "")

                    if isinstance(summary_text, pd.Series):
                        summary_text = summary_text.iloc[0]

                    if pd.notna(summary_text) and str(summary_text).strip() != "":
                        with st.expander("Read more"):
                            st.write(summary_text)

                    hazard_text = row.get("Known Hazards", "")

                    if isinstance(hazard_text, pd.Series):
                        hazard_text = hazard_text.iloc[0]

                    if (
                        pd.notna(hazard_text)
                        and str(hazard_text).strip() != ""
                        and str(hazard_text).lower() != "none known"
                    ):
                        with st.expander("Safety details"):
                            st.write("**Please review this safety information carefully:**")
                            st.write(hazard_text)
