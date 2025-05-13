import streamlit as st
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, pipeline

# Force PyTorch to use CPU
device = torch.device('cpu')

# Load Toxic BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device)

# Load QA model
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def parse_whatsapp_chat(uploaded_file):
    lines = uploaded_file.getvalue().decode('utf-8').splitlines()
    chat = []
    for line in lines:
        match = re.match(r"\[(\d{1,2}/\d{1,2}/\d{2,4}), ([\d:]+)\s?(AM|PM|am|pm)?\] (.*?): (.*)", line)
        if match:
            date, time, meridiem, sender, message = match.groups()
            time = f"{time} {meridiem}" if meridiem else time
            chat.append({"date": date, "time": time, "sender": sender, "message": message})
    return pd.DataFrame(chat)

def classify_messages(df):
    df['clean_message'] = df['message'].apply(clean_text)

    # Tokenize the messages
    inputs = tokenizer(df['clean_message'].tolist(), truncation=True, padding=True, max_length=100, return_tensors="pt").to(device)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy()  # shape: (num_samples, 6)

    # Apply threshold of 0.5
    labels = (probs >= 0.5).astype(int)

    # Label names from toxic-bert
    toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    for i, label in enumerate(toxic_labels):
        df[label] = labels[:, i]

    df['any_toxic'] = (labels.sum(axis=1) > 0).astype(int)
    return df

# Streamlit UI
st.title("ThreatSync")

uploaded_file = st.file_uploader("Upload your WhatsApp chat text file", type=["txt"])

if uploaded_file:
    chat_df = parse_whatsapp_chat(uploaded_file)
    if chat_df.empty:
        st.error("Failed to parse chat. Please check the format.")
    else:
        st.subheader("Parsed Chat")
        st.dataframe(chat_df.head())

        # Classify toxic messages
        chat_df = classify_messages(chat_df)

        st.subheader("Detected Toxic Messages")

        # Extract only messages with any toxicity
        toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        toxic_messages = []

        for _, row in chat_df.iterrows():
            active_labels = [label for label in toxic_labels if row[label] == 1]
            if active_labels:
                toxic_messages.append({
                    "Sender": row['sender'],
                    "Message": row['message'],
                    "Toxic Categories": ", ".join(active_labels)
                })

        if toxic_messages:
            st.markdown("### ⚠️ Toxic Messages Identified")
            for msg in toxic_messages:
                st.markdown(f"**{msg['Sender']}**: {msg['Message']}  \n*Categories: {msg['Toxic Categories']}*")
        else:
            st.success("No toxic messages found! 🎉")

        st.subheader("Ask Questions About the Chat")
        user_question = st.text_input("Enter your question:")
        if user_question:
            context = " ".join(chat_df['message'].astype(str))
            try:
                result = qa_pipeline(question=user_question, context=context)
                st.write(f"**Answer:** {result['answer']}")
            except Exception as e:
                st.error(f"QA failed: {e}")