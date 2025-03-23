# Language model decoder analysis

This folder contains the code for the language model (LM) decoder analysis. 

Raw DrugBank outcome descriptions can be extracted from the data we provided using the first section of `./process_input_data.ipynb`. 

To generate augmented textual descriptions, please follow instructions in `text_augmentation/openai_api_batch_creation_processing.ipynb` (including running `text_augmentation/openai_api_request_parallel_processor.py`), which will generate `./api_requests_multi_new.jsonl` and `./api_requests_results_multi_new.jsonl`. Then, data for training LM decoder can be processed with the second section of `./process_input_data.ipynb`.

After generating data, we are now ready to train the model with `train_ddi_mistral.py`. 
