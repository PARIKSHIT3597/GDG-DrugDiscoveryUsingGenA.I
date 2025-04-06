import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle

# ------------------------------
# Molecular descriptor calculator
# ------------------------------
def desc_calc():
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')

# ------------------------------
# File download helper
# ------------------------------
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# ------------------------------
# Model prediction
# ------------------------------
def build_model(input_data, load_data):
    model_path = os.path.join("models", "acetylcholinesterase_model.pkl")
    with open(model_path, "rb") as f:
        load_model = pickle.load(f)

    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# ------------------------------
# Display Logo
# ------------------------------
try:
    image = Image.open("logo.png")  # Make sure logo.png is in the same folder as app.py
    st.image(image, use_column_width=True)
except FileNotFoundError:
    st.warning("⚠️ Logo not found. Please make sure 'logo.png' is present.")

# ------------------------------
# Page Title
# ------------------------------
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This app allows you to predict the bioactivity towards inhibiting the `Acetylcholinesterase` enzyme. `Acetylcholinesterase` is a drug target for Alzheimer's disease.
""")

# ------------------------------
# File Upload Section
# ------------------------------
with st.sidebar.header('1. Upload your input file (.txt format)'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

# ------------------------------
# Main Execution Block
# ------------------------------
if st.sidebar.button('Predict'):
    if uploaded_file is not None:
        load_data = pd.read_table(uploaded_file, sep=' ', header=None)
        load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

        st.header('**Original input data**')
        st.write(load_data)

        with st.spinner("Calculating descriptors..."):
            desc_calc()

        # Display calculated descriptors
        st.header('**Calculated molecular descriptors**')
        desc = pd.read_csv("descriptors_output.csv")
        st.write(desc)
        st.write(desc.shape)

        # Load descriptor list
        st.header('**Subset of descriptors from previously built model**')
        descriptor_list_path = os.path.join("models", "descriptor_list.csv")
        Xlist = list(pd.read_csv(descriptor_list_path).columns)
        desc_subset = desc[Xlist]
        st.write(desc_subset)
        st.write(desc_subset.shape)

        # Predict
        build_model(desc_subset, load_data)
    else:
        st.warning("⚠️ Please upload a valid `.txt` file to proceed.")
else:
    st.info('👈 Upload input data in the sidebar and click **Predict** to begin.')
