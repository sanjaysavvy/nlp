import streamlit as st
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device)

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

    
    inputs = tokenizer(df['clean_message'].tolist(), truncation=True, padding=True, max_length=100, return_tensors="pt").to(device)

    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy()  

    
    labels = (probs >= 0.5).astype(int)

    
    toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    for i, label in enumerate(toxic_labels):
        df[label] = labels[:, i]
        df[label + '_prob'] = probs[:, i]  

    df['any_toxic'] = (labels.sum(axis=1) > 0).astype(int)
    return df


st.title("ThreatSync")

uploaded_file = st.file_uploader("Upload your WhatsApp chat text file", type=["txt"])

if uploaded_file:
    chat_df = parse_whatsapp_chat(uploaded_file)
    if chat_df.empty:
        st.error("Failed to parse chat. Please check the format.")
    else:
        
        chat_df = classify_messages(chat_df)

        
        threat_msgs = chat_df[chat_df['threat'] == 1]
        if not threat_msgs.empty:
            first_threat = threat_msgs.iloc[0]
            st.markdown(f"### 🚨 First Threatening Message")
            st.markdown(f"**{first_threat['sender']}** said: *{first_threat['message']}*")
        else:
            st.markdown("### ✅ No threats detected in the conversation.")

        st.subheader("Parsed Chat")
        st.dataframe(chat_df.head())

        st.subheader("Detected Toxic Messages")

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
            st.markdown("### Toxic Messages Identified")
            for msg in toxic_messages:
                st.markdown(f"**{msg['Sender']}**: {msg['Message']}  \n*Categories: {msg['Toxic Categories']}*")
        else:
            st.success("No toxic messages found! 🎉")

        
        st.subheader("Explore by Participant")
        participants = chat_df['sender'].unique()
        col1, col2 = st.columns(2)
        with col1:
            for person in participants[:len(participants)//2 + len(participants)%2]:
                if st.button(person, key=f"user_{person}"):
                    filtered = chat_df[chat_df['sender'] == person]
                    st.markdown(f"### Messages by **{person}**")
                    for _, row in filtered.iterrows():
                        st.markdown(f"- {row['message']}")
        with col2:
            for person in participants[len(participants)//2 + len(participants)%2:]:
                if st.button(person, key=f"user_{person}"):
                    filtered = chat_df[chat_df['sender'] == person]
                    st.markdown(f"### Messages by **{person}**")
                    for _, row in filtered.iterrows():
                        st.markdown(f"- {row['message']}")

        
        st.subheader("Explore by Toxic Category")
        col3, col4 = st.columns(2)
        with col3:
            for label in toxic_labels[:3]:
                if st.button(label, key=f"label_{label}"):
                    filtered = chat_df[chat_df[label] == 1]
                    st.markdown(f"### Messages labeled as **{label}**")
                    for _, row in filtered.iterrows():
                        prob = row[label + '_prob']
                        st.markdown(f"**{row['sender']}**: {row['message']}  \n*Confidence: {prob:.2f}*")
        with col4:
            for label in toxic_labels[3:]:
                if st.button(label, key=f"label_{label}"):
                    filtered = chat_df[chat_df[label] == 1]
                    st.markdown(f"### Messages labeled as **{label}**")
                    for _, row in filtered.iterrows():
                        prob = row[label + '_prob']
                        st.markdown(f"**{row['sender']}**: {row['message']}  \n*Confidence: {prob:.2f}*")