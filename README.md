# CircEvent
This repository contains the source code for CircEvent, which incorporates circumstance into narrative event prediction.
The source code is divided into two parts, i.e. data preprocessing and model training.
The script is placed in `bin` folder. Each step and its corresponding script is listed below.

# Reproduce Steps
1. extract text out of gigaword xml file. `1-extract_gigaword_nyt`
2. annotate text with CoreNLP. `2-corenlp_annotate`
3. extract event chain from annotated document. `3-extract_event_chain`
4. convert event chain words to ids. `4-index_event_chain`
5. split into train, validation, test set. `5-split_dataset`
6. train the circ model. `6-circ_train`
7. evaluate the saved model. `7-circ_eval`

# Environment & Setup
We conducted our experiments with on a workstation with a RTX 2080Ti, 64GB Memory.
Our programs are tested under PyTorch 1.8.1 + CuDA 10.2.  
You can follow these steps to reproduce our experiments.

1. Setup Python environment. We encourage using conda to setup the python virtual environment.
`conda create -n circ python==3.8`
2. Install the CuDA toolkit and Pytorch.
`conda install cudatoolkit=10.2`
3. Install the pip packages.
`pip install -r requirements.txt`
4. Install 

# Dataset
For 