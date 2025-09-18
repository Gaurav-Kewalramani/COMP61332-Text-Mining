
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BERTForRelationExtraction(nn.Module):
    def __init__(self, num_labels, pretrained="bert-base-cased"):
        super(BERTForRelationExtraction, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained)
        self.hidden_size = self.bert.config.hidden_size

        # Entity-aware attention
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)

        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # [CLS] embedding

        # Entity-aware attention using CLS as query
        cls_expanded = cls_output.unsqueeze(1)
        attended_output, _ = self.entity_attention(
            cls_expanded,
            sequence_output,
            sequence_output,
            key_padding_mask=(attention_mask == 0)
        )
        attended_output = attended_output.squeeze(1)

        # Combine [CLS] and attention result
        combined = torch.cat([cls_output, attended_output], dim=1)
        combined = self.dropout(combined)

        # Classification
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}
