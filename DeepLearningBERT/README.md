# BERT Relation Extraction Model

This model performs relation extraction on sentences containing marked entities.

## Model Links

https://livemanchesterac-my.sharepoint.com/:u:/g/personal/mohammedismaelabidali_shaikh_postgrad_manchester_ac_uk/Ed9iif_1365BnCf5AuUMO8kBkQWxpfj3eq4UtUTOH0gY1g?e=M0rCah

Larger Model:
https://livemanchesterac-my.sharepoint.com/:u:/g/personal/mohammedismaelabidali_shaikh_postgrad_manchester_ac_uk/Ef0orOcEUOpAj_L3083MpycB6ksex5_81KFv0CuYWie24g?e=exlsCW

## Model Details

- Architecture: BERT with entity-aware attention mechanism
- Dataset: RE-TACRED
- Number of relation classes: 40
- Entity types: CAUSE_OF_DEATH, CITY, COUNTRY, CRIMINAL_CHARGE, DATE, DURATION, IDEOLOGY, LOCATION, NATIONALITY, NUMBER, ORGANIZATION, PERSON, RELIGION, STATE_OR_PROVINCE, TITLE, URL
- Maximum sequence length: 160

## Inference Mode

## Usage

```python
from predict import load_model, predict_relation

# Load the model
model, tokenizer, metadata = load_model("./bert_relation_extraction")

# Example sentence with entities
sentence = "John Smith is the CEO of Microsoft Corporation."
subject_start, subject_end = 0, 1  # "John Smith"
object_start, object_end = 6, 7    # "Microsoft Corporation"

# Predict relation
result = predict_relation(
   model, tokenizer,
   sentence,
   subject_start, subject_end, "PERSON",
   object_start, object_end, "ORGANIZATION"
)

print(f"Predicted relation: {result['relation']} (confidence: {result['confidence']:.4f})")
```

## Alternate Inference
The notebook can be loaded up in google colab, and the model can also be loaded up in google colab. Followingly, the demo() function can be used to run the notebook.

## Files

- `model_state.pt`: Model weights
- `metadata.json`: Model metadata including label mappings
- `model.py`: Model architecture definition
- `predict.py`: Prediction function
- `special_tokens_map.json`, `tokenizer_config.json`, `vocab.txt`: Tokenizer files

## Evaluation Results

- Micro F1: 0.9021
- Macro F1: 0.7347
- Filtered F1 (excluding no_relation): 0.7890

## USE of Generative AI

In this codebase, generative AI was primarily used for generating the demo code for inference, so that one can test 
their examples. Followingly, it was used to generate the examples in the demo loop once again. It was used to clean up irrelevant/leftover code. 
Google colab's gemini was utilized for help with errors. Furthermore, it was used for help within the logits part/mixup part due to unfamiliarity with the subject. 


