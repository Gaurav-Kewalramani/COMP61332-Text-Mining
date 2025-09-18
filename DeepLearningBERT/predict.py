
import torch
import json
import torch.nn.functional as F
from transformers import BertTokenizer
from model import BERTForRelationExtraction

def load_model(model_dir):
    # Load metadata
    with open(f"{model_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)

    # Get model parameters
    num_labels = metadata.get('num_labels', 40)

    # Initialize model
    model = BERTForRelationExtraction(num_labels=num_labels)

    # Load model state
    model.load_state_dict(torch.load(f"{model_dir}/model_state.pt",
                                    map_location=torch.device('cpu')))

    # Load tokenizer with special tokens
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Set model to evaluation mode
    model.eval()

    return model, tokenizer, metadata

def predict_relation(model, tokenizer, sentence, subject_start, subject_end, subject_type,
                    object_start, object_end, object_type, max_length=256, device='cpu'):
    # Load id2label mapping
    with open("metadata.json", 'r') as f:
        metadata = json.load(f)
        id2label = metadata.get('id2label', {})

    # Convert string keys to integers if needed
    if all(k.isdigit() for k in id2label.keys()):
        id2label = {int(k): v for k, v in id2label.items()}

    # Tokenize the sentence
    tokens = sentence.split()

    # Insert entity markers
    if subject_end < object_start:
        # Subject comes before object
        tokens.insert(subject_start, f'[S:{subject_type}]')
        tokens.insert(subject_end + 2, f'[/S:{subject_type}]')
        object_start += 2
        object_end += 2
        tokens.insert(object_start, f'[O:{object_type}]')
        tokens.insert(object_end + 2, f'[/O:{object_type}]')
    else:
        # Object comes before subject
        tokens.insert(object_start, f'[O:{object_type}]')
        tokens.insert(object_end + 2, f'[/O:{object_type}]')
        subject_start += 2
        subject_end += 2
        tokens.insert(subject_start, f'[S:{subject_type}]')
        tokens.insert(subject_end + 2, f'[/S:{subject_type}]')

    # Join tokens
    marked_text = " ".join(tokens)

    # Encode for BERT
    encoding = tokenizer(marked_text,
                        truncation=True,
                        max_length=max_length,
                        padding='max_length',
                        return_tensors='pt')

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
        probs = F.softmax(logits, dim=1)

        # Get the most likely relation
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        relation = id2label[pred_idx]

        # Get top 3 predictions
        top3_values, top3_indices = torch.topk(probs[0], 3)
        top3_preds = [(id2label[idx.item()], val.item()) for idx, val in zip(top3_indices, top3_values)]

    return {
        'relation': relation,
        'confidence': confidence,
        'top3': top3_preds
    }

# Example usage:
# model, tokenizer, metadata = load_model("./bert_relation_extraction")
# result = predict_relation(
#    model, tokenizer,
#    "John Smith is the CEO of Microsoft Corporation.",
#    0, 1, "PERSON", 6, 7, "ORGANIZATION"
# )
# print(f"Predicted relation: {result['relation']} (confidence: {result['confidence']:.4f})")
