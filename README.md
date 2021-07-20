# CircEvent

# Pre-process
1. extract text out of gigaword xml file. `extract_gigaword_nyt`
2. annotate text with CoreNLP. `corenlp_annotate`
3. extract event chain from annotated document. `extract_event_chain`
4. convert event chain words to ids. `index_event_chain`
5. split into train, validation, test set. `split_dataset`

