import streamlit as st
from transformers import pipeline

#load models
senti_model=pipeline("sentiment-analysis",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
gener_model=pipeline("text-generation", model="openai/gpt-oss-120b")

st.title("BERT and GPT models")
st.write("I have made GPT Sentiment Analysis using BERT model and  Text Generation using GPT and streamlit application  ")
menu=st.sidebar.selectbox("Choose a model",["Sentiment Analysis", "Text Generation"])

if menu=="Sentiment Analysis":
  st.header("BERT Model")
  text=st.text_area("Enter a sentence: ")
  if st.button("Analyze sentiment"):
    result=senti_model(text)[0]
    st.success("Prediction: ",result['label'])

  else:
    st.header("GPT Model")
    prompt=st.text_area("Enter a prompt: ")
    if st.button("enerate Text"):
      output=gener_model(prompt, max_length=80)
      st.write(output[0])["generated_text"]
