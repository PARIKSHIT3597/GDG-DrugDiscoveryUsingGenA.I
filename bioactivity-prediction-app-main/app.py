import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from PIL import Image
import base64
import pickle
import os

# RDKit-based descriptor calculator (Morgan Fingerprints)
def desc_calc_rdkit(smiles_list, mol_names):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) if mol else None
        for mol in mols
    ]
    valid_data = [(name, fp) for name, fp in zip(mol_names, fingerprints) if fp is not None]
    if not valid_data:
        return None
    names, fps = zip(*valid_data)
    arr = np.array([list(fp.ToBitString()) for fp in fps], dtype=int)
    df = pd.DataFrame(arr)
    df.insert(0, 'molecule_name', names)
    df.to_csv('descriptors_output.csv', index=False)
    return df

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model prediction
def build_model(input_data, molecule_names):
    model = pickle.load(open('acetylcholinesterase_model.pkl', 'rb'))
    prediction = model.predict(input_data)
    st.header('**Prediction Output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(molecule_names, name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Load and display logo (optional)
if os.path.exists("logo.png"):
    image = Image.open('logo.png')
    st.image(image, use_container_width=True)

# Title
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This app predicts bioactivity towards inhibiting the **Acetylcholinesterase** enzyme.  
A key target in drug discovery for Alzheimer's treatment.

---
""")

# Sidebar
with st.sidebar.header('1. Upload your SMILES file (.txt)'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

# Prediction
if st.sidebar.button('Predict'):
    if uploaded_file is not None:
        load_data = pd.read_table(uploaded_file, sep=' ', header=None)
        smiles_list = load_data[0].tolist()
        molecule_names = load_data[1].tolist()

        st.header('**Original Input Data**')
        st.write(load_data)

        with st.spinner("Calculating descriptors..."):
            desc = desc_calc_rdkit(smiles_list, molecule_names)

        if desc is None:
            st.error("Descriptor calculation failed. Please check your SMILES strings.")
        else:
            st.header('**Calculated Molecular Descriptors**')
            st.write(desc)
            st.write(desc.shape)

            # Subset descriptor logic
            st.header('**Subset of Descriptors Used in Model**')
            try:
                Xlist = list(pd.read_csv('descriptor_list.csv').columns)
                desc_subset = desc[Xlist]
            except FileNotFoundError:
                st.warning("descriptor_list.csv not found. Using all descriptors.")
                desc_subset = desc.drop('molecule_name', axis=1)

            st.write(desc_subset)
            st.write(desc_subset.shape)

            build_model(desc_subset, desc['molecule_name'])
    else:
        st.warning('Please upload a file to continue.')
else:
    st.info('Upload input data in the sidebar to start.')
