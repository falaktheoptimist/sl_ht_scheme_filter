import os
from ast import literal_eval

import pandas as pd
import streamlit as st
from langchain.llms import Clarifai
from langchain import PromptTemplate
from langchain.document_loaders import BSHTMLLoader


st.set_page_config(layout="centered")

@st.cache_data
def load_pat():
    if "CLARIFAI_PAT" not in st.secrets:
        st.error("You need to set the CLARIFAI_PAT in the secrets.")
        st.stop()
    return st.secrets.CLARIFAI_PAT


@st.cache_resource
def get_text_model(pat):
    USER_ID = "openai"
    APP_ID = "chat-completion"
    MODEL_ID = "GPT-4"
    llm = Clarifai(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
    return llm


def read_html_files(html_paths):
    docs = []
    for path in html_paths:
        loader = BSHTMLLoader(path)
        doc = loader.load()[0]
        lines = doc.page_content.split("\n")
        filtered_lines = [line for line in lines if line.strip()]
        docs.append("\n".join(filtered_lines))
    return docs


def get_prompt(text):
    prompt_template = """Analyze the provided text to extract eligibility criteria for 
the following fields: Gender, Occupations, and Category, based on information from a 
government scheme. The eligible options for each field are as follows:
1. Gender: ["Male", "Female", "Others", "None"]
2. Occupations: ["Student", "Farmers", "Retired", "None"]
3. Category: ["SC", "ST", "OBC", "General", "None"]

Text to Analyze: {text}


Output Format: Only return a json object wrapped around by "<json> </json>" tags

Example: <json>{{"gender": ["value_1", "value_2", ...], "occupation": ["value_1", "value_2", ...], "category": ["value_1", "value_2", ...]}}</json>

Note: 
* Ensure that the extracted values for the eligibility criteria are from the provided options only.

Response:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    return prompt.format(text=text)


@st.cache_data
def load_database():
    df = pd.read_csv("database.csv", converters={
        "gender": literal_eval,
        "occupation": literal_eval,
        "category": literal_eval
    })
    return df

pat = load_pat()
llm = get_text_model(pat)
df = load_database()


st.header("`Goverment Schemes Explorer`")
st.info("`I am an AI that can given information about different government schemes applicable to you.`")


st.subheader('User Information', divider="green")
with st.form("my_form"):
    gender = st.selectbox(
        "Gender: ",
        ("Male", "Female", "Others")
    )
    occupation = st.selectbox(
        "Occupation: ",
        ("Student", "Farmers", "Retired")
    )
    category = st.selectbox(
        "Gender: ",
        ("SC", "ST", "OBC", "General")
    )
 
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

if submitted:
    st.subheader('Schemes', divider="green")
    st.write(gender, occupation, category)
    filtered_df = df.loc[
        df.apply(
            lambda x: gender in x["gender"] or "None" in x["gender"], 
            axis=1
        )
    ]
    filtered_df = filtered_df.loc[
        filtered_df.apply(
            lambda x: occupation in x["occupation"] or "None" in x["occupation"], 
            axis=1
        )
    ]
    filtered_df = filtered_df.loc[
        filtered_df.apply(
            lambda x: category in x["category"] or "None" in x["category"], 
            axis=1
        )
    ]
    st.dataframe(filtered_df)
