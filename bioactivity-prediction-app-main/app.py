import streamlit as st
import pandas as pd
import base64
import pickle
import os
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from PIL import Image

# Load logo
logo_path = os.path.join("logo.png")
image = Image.open(logo_path)
st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This app predicts the bioactivity for inhibiting the `Acetylcholinesterase` enzyme using RDKit-generated molecular descriptors.
""")

# Descriptor Calculation with RDKit
def calculate_descriptors(smiles_list):
    descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    descriptors = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            descriptors.append(calc.CalcDescriptors(mol))
            valid_smiles.append(smi)
        else:
            descriptors.append([None]*len(descriptor_names))
            valid_smiles.append(None)

    df = pd.DataFrame(descriptors, columns=descriptor_names)
    return df, valid_smiles

# Download link
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Build model
def build_model(input_data, molecule_names):
    model_path = os.path.join("models", "acetylcholinesterase_model.pkl")
    with open(model_path, "rb") as f:
        load_model = pickle.load(f)

    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    mol_name = pd.Series(molecule_names, name='molecule_name')
    df = pd.concat([mol_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Sidebar
with st.sidebar.header('1. Upload your SMILES file'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

if st.sidebar.button('Predict'):
    if uploaded_file is not None:
        load_data = pd.read_table(uploaded_file, sep=' ', header=None)
        st.header('**Original input data**')
        st.write(load_data)

        smiles_list = load_data[0].tolist()
        descriptor_df, valid_smiles = calculate_descriptors(smiles_list)

        st.header('**Calculated molecular descriptors**')
        st.write(descriptor_df)
        st.write(descriptor_df.shape)

        # Read descriptor list used in model
        descriptor_list_path = os.path.join("models", "descriptor_list.csv")
        selected_descriptors = list(pd.read_csv(descriptor_list_path).columns)
        desc_subset = descriptor_df[selected_descriptors]

        st.header('**Subset of descriptors used in the model**')
        st.write(desc_subset)

        build_model(desc_subset, load_data[1])
    else:
        st.warning("Please upload a file.")
else:
    st.info('Upload input data in the sidebar to start!')
