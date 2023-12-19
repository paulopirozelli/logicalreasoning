import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput

# For reproduction, set seed
import torch
torch.manual_seed(1234)

class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super().__init__()
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True)

        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint, config=config)

        self.classifier = nn.Linear(config.hidden_size, num_labels)  # load and initialize weights
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        x = self.dropout(outputs[0][:, 0, :])  # outputs[0]=last hidden state
        logits = self.classifier(x)  # calculate losses

        return TokenClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions), logits, logits