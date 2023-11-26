import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util

# Load your FinBERT model
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
tokenizer_finbert = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp_finbert = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer_finbert)

# Load your Sentence Transformer model
model_st = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

@st.cache_resource
def get_model():
    # Load your BERT model pretrained by fake news detection
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertForSequenceClassification.from_pretrained("C:/Users/oka resia/Downloads/NLP/Fake news detection/pretrained model/fakenews_BERT")

    return tokenizer_bert, model_bert

tokenizer_bert, model_bert = get_model()

st.write("""
# News Sentiment Analysis for Company
""")

headline_input = st.text_area('Enter Headline to Analyze')
body_input = st.text_area('Enter Company Profile')

button_bert = st.button("Analyze with **BERT**")
button_finbert = st.button("Analyze with **FinBERT**")

ids_to_labels = {0: "unrelated", 1: "neutrally_related", 2: "negatively_related", 3: "positively_related"}

if headline_input and body_input and button_finbert:
    # Perform FinBERT sentiment analysis
    tweets_finbert = [headline_input + ' ' + body_input]
    results_finbert = nlp_finbert(tweets_finbert)
    sentiment_finbert = results_finbert[0]['label']
    st.write("Sentiment (FinBERT): ", sentiment_finbert)

if headline_input and body_input and button_finbert:
    # Perform cosine similarity calculation using FinBERT embeddings
    similarity_score_finbert = util.pytorch_cos_sim(
        model_st.encode(headline_input),
        model_st.encode(body_input)
    ).item()
    st.write("Similarity Score (FinBERT): ", similarity_score_finbert)
    
if headline_input and body_input and button_bert:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model_bert = model_bert.cuda()

    data_bert = {'articleBody': [body_input], 'Headline': [headline_input]}
    user_input_bert = pd.DataFrame(data_bert)
    test_sample_bert = tokenizer_bert(user_input_bert.values.tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')

    input_id_bert = torch.tensor(test_sample_bert['input_ids']).to(device)
    mask_bert = torch.tensor(test_sample_bert['attention_mask']).to(device)

    output_bert = model_bert(input_id_bert, mask_bert, None)
    st.write("Logits (BERT): ", output_bert.logits)
    y_pred_bert = np.argmax(output_bert.logits.detach().numpy(), axis=1)
    st.write("Prediction (BERT): ", ids_to_labels[y_pred_bert[0]])


