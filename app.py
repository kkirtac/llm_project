import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("train/finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("train/finetuned_model")

st.title("Text to Sequence of Nodes")
prompt = st.text_input("Enter a prompt:", "")

# Debug print for prompt
st.write(f"Prompt entered: {prompt}")

if st.button("Generate Nodes"):
    if prompt:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Generated Sequence of Nodes:")
        st.write(generated_text)
    else:
        st.write("Please enter a prompt to generate nodes.")

