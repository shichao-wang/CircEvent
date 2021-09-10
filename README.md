# CircEvent
This repository contains the source code for the publication of EMNLP 2021 findings *Incorporating Circumstance into Narrative Event Prediction*.
You can follow the instructions below to reproduce our experiments.

## Dataset
Our experiments are conducted on the New York Times (NYT) portion of the English Gigawords.
You can get access from the [official website](https://catalog.ldc.upenn.edu/LDC2003T05).
The data split we used is provided by Granroth-Wilding[[1]](https://mark.granroth-wilding.co.uk/papers/what_happens_next/)
We annotate the raw documents based on Lee[[2]](https://github.com/doug919/multi_relational_script_learning) with the standford CoreNLP toolkit.
The configuration of CoreNLP is listed in `corenlp.props` file.

## Environment Setup
We conducted our experiments with on a workstation with a RTX 2080Ti, 64GB Memory.
Our programs are tested under PyTorch 1.8.1 + CUDA 10.2.

1. Setup Python environment. We encourage using conda to setup the python virtual environment.
   `conda create -n circ python==3.8 && conda activate circ`
2. Install the CUDA toolkit and Pytorch.
   `conda install cudatoolkit=10.2 && pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html`
4. Install the pip packages.
   `pip install -r requirements.txt`
5. Install the circumst_event package
   `pip install -e .`

Now the environment has been set up in the `circ` virtual environment.

## Reproduce Steps
The source codes can be divided into two parts, i.e. data preprocessing and model training.
The entry scripts are placed in `bin` folder. Each step and its corresponding script is listed below.

1. extract text out of gigaword xml file. `1-extract_gigaword_nyt`
2. annotate text with CoreNLP. `2-corenlp_annotate`
3. extract event chain from annotated document. `3-extract_event_chain`
4. convert event chain words to ids. `4-index_event_chain`
5. split into train, validation, test set. `5-split_dataset`
6. train the circ model. `6-circ_train`
7. evaluate the saved model. `7-circ_eval`


# Reference
[1] Mark Granroth-Wilding and Stephen Clark. 2016. What happens next? Event Prediction Using a Compositional Neural Network Model. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, pages 2727–2733, Phoenix, Arizona, February. AAAI Press.

[2] I-Ta Lee and Dan Goldwasser. 2019. Multi-Relational Script Learning for Discourse Relations. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4214–4226, Florence, Italy, July. Association for Computational Linguistics.