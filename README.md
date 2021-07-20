# CircEvent
This repository contains the source code for CircEvent, which incorporates circumstance into narrative event prediction.
The source code is divided into two parts, i.e. data preprocessing and model training.
The script is placed in `src` folder. Each step and its corresponding script is listed below.
# Steps
1. extract text out of gigaword xml file. `extract_gigaword_nyt`
2. annotate text with CoreNLP. `corenlp_annotate`
3. extract event chain from annotated document. `extract_event_chain`
4. convert event chain words to ids. `index_event_chain`
5. split into train, validation, test set. `split_dataset`
6. train the circ model. `circ_train`
7. evaluate the saved model. `circ_eval`

# Environment
We conducted our experiments with on a workstation with Dual RTX 2080Tis, 64GB Memory.
Our program is tested under PyTorch 1.8.1 + CuDA 10.2.