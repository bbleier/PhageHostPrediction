import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init

import numpy as np
import pandas as pd

from transformers import AutoModel, AutoTokenizer


class DNABertModel(nn.Module):
    """
    A modified BERT model that accepts pre-computed embeddings instead of token IDs and
    adds a CLS token embedding at the beginning of each sequence. This model is designed
    to process embeddings directly and handles the BERT encoding with an updated attention
    mechanism to include the CLS token.

    Inputs:
        chunk_embeddings: A tensor of shape (m, n, d), where m is the number of DNA samples,
                          n is the number of chunks per sample, and d is the embedding dimension.
                          These embeddings represent DNA chunks that have been pre-computed.
        attention_mask: A tensor of shape (m, n) that indicates which chunks are actual DNA data
                        and should be processed by the model. This mask will be updated internally
                        to reflect the addition of the CLS token at the start of each sequence.

    Outputs:
        Tensor of shape (m, n+1, 768): The output embeddings from the last layer of the BERT encoder,
        including the CLS token at the beginning of each sequence. Each embedding corresponds to
        a chunk of DNA or the CLS token, and the sequence has been processed to integrate information
        across the chunks.

    Parameters:
        pretrained_bert_model: An instance of a pre-trained BERT model. This model will be adapted
                               to accept embeddings directly.
        cls_token_id: The token ID used for the CLS token in the original BERT vocabulary. This ID
                      is used to retrieve the CLS token embedding from the pre-trained model's embedding
                      layer.
    """
    def __init__(self, pretrained_bert_model, cls_token_id=1):
        super(DNABertModel, self).__init__()
        # Load a pre-trained BERT model
        self.bert_model = pretrained_bert_model

        # Get the CLS token embedding.  This will be added to the start of the DNA chunk embeddings
        # Note: Need to clone and detach or it causes break in computational graph and backprop will fail
        self.cls_embedding = self.bert_model.embeddings.word_embeddings(torch.tensor([cls_token_id], device=pretrained_bert_model.device)).clone().detach().requires_grad_(True)

        # Note: Positional embeddings not needed as base DNABERT2 model uses ALiBi to include positional info (Attention with Linear Biases)

    def forward(self, chunk_embeddings, attention_mask):
        # Expand CLS embeddings to match the batch size and concatenate it to the beginning of chunk embeddings.
        # This will result in CLS embeddings being the first embeddings for each DNA strand in a given batch
        # of DNA chunks
        batch_size = chunk_embeddings.size(0)
        cls_embeddings = self.cls_embedding.expand(batch_size, 1, -1) #mx1x768
        embeddings_with_cls = torch.cat([cls_embeddings, chunk_embeddings], dim=1) #mx(n+1)x768

        # Update the attention mask to include the CLS token
        cls_attention_mask = torch.ones((batch_size, 1), device=embeddings_with_cls.device, dtype=torch.long) #mx1
        attention_mask_with_cls = torch.cat([cls_attention_mask, attention_mask], dim=1) #mx(n+1)

        # Capture the sequence length to maintain batch shape later on
        sequence_lengths = attention_mask_with_cls.sum(dim=1).tolist()

        # Directly input embeddings to the encoder to bypass embedding step
        # Note: This requires some manual post processing due to the way in which the
        # model outputs the results.  It will only output samples for which the attention
        # mask was =1.  It will then concatenate all results into one dimension.  Need to chunk
        # up the results and pad to maintiain batch dimension

        encoder_outputs = self.bert_model.encoder(
            embeddings_with_cls,
            attention_mask=attention_mask_with_cls)
        
        #Encoder ouputs all layers of model.  Pull the last layer only.
        encoder_outputs = encoder_outputs[-1]

        #Post process the results
        encoder_outputs = self.post_process_output(encoder_outputs, sequence_lengths)

        return encoder_outputs

    def post_process_output(self, encoder_outputs, sequence_lengths):
        '''Takes the encoder outputs and the length of each sequence of non-padded embeddings
        and breaks up into chunks where each chunk is a sample in the batch.  Will return
        a matrix of size [mxnx768] where m is the number of samples in the batch, n is the
        maximum number of attention mask values = 1, and 768 is the embedding dimension'''

        # Prepare to break up the outputs to retain batch dimension
        segmented_outputs = []
        start_index = 0
        for length in sequence_lengths:
            end_index = start_index + length
            segmented_outputs.append(encoder_outputs[start_index:end_index])
            start_index = end_index

        #(TODO): NO NEED TO REPAD.  JUST TAKE CLS TOKENS AND OUTPUT DIRECTLY

        # Pad the sequences so they all have the same length
        padded_outputs = pad_sequence(segmented_outputs, batch_first=True, padding_value=0)

        return padded_outputs

class CustomClassifier(nn.Module):
    '''A compiler for a custom DNABERT-2 model that will input CLS token embeddings outputted
    from another DNABERT-2 model, created by chunking DNA strands into 512 token chunks, and
    will output a tensor of logits including a CLS logit that contains context from the entire
    DNA strand.

    Inputs:
        input_ids: A tensor of shape (m, n, 768), where m is the number of DNA samples,
                   n is the number of chunks per sample, and 768 is the embedding dimension.
        attention_mask: A tensor of shape (m, n) that indicates which tokens are padding
                        and should not be processed by the model.

    Outputs:
        A tensor of shape (m, num_classes), where each row contains the predicted logits
        for each class for a DNA sample. The model aggregates information across chunks
        per sample to produce a single prediction vector per sample.
    '''
    def __init__(self, dna_bert_model, protein_model, num_classes, hidden_layer_sizes=[], cls_token_id=1, dropout=0.5, init_mode=''):
        super(CustomClassifier, self).__init__()
        #Load the custom DNA BERT Model
        self.bert = DNABertModel(dna_bert_model, cls_token_id=cls_token_id)

        # Setup initial input size
        input_size = dna_bert_model.config.hidden_size + protein_model.config.hidden_size

        # Create hidden layers if inputted
        modules = []
        for hidden_layer_size in hidden_layer_sizes:
            layer = nn.Linear(input_size, hidden_layer_size)
            if init_mode == 'kaiming_normal':
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif init_mode == 'kaiming_uniform':
                init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
            modules.append(layer)
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout)) 
            input_size = hidden_layer_size

        # Add final classification head and group modules together
        modules.append(nn.Linear(input_size, num_classes))
        self.classifier = nn.Sequential(*modules)

        #softmax the output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, DNA_chunk_embeddings, DNA_attention_mask, protein_embeddings, return_probs=True):
        #Process through the custom BERT model
        dna_bert_outputs = self.bert(DNA_chunk_embeddings, attention_mask=DNA_attention_mask)

        # Extract the embeddings of the CLS token and run through classifier
        dna_embeddings = dna_bert_outputs[:, 0, :] # Take the CLS token from custom DNABERT model

         # Combine outputs
        combined_embeddings = torch.cat([dna_embeddings, protein_embeddings], dim=-1)

        # Perform classification
        logits = self.classifier(combined_embeddings)

        #Determine if probabilities are a desired output
        if return_probs:
            probs = self.softmax(logits)
            return probs
        else:
            return logits

class DNAOnlyClassifier(nn.Module):
    '''A compiler for a custom DNABERT-2 model that will input CLS token embeddings outputted
    from another DNABERT-2 model, created by chunking DNA strands into 512 token chunks, and
    will output a tensor of logits including a CLS logit that contains context from the entire
    DNA strand.

    Inputs:
        input_ids: A tensor of shape (m, n, 768), where m is the number of DNA samples,
                   n is the number of chunks per sample, and 768 is the embedding dimension.
        attention_mask: A tensor of shape (m, n) that indicates which tokens are padding
                        and should not be processed by the model.

    Outputs:
        A tensor of shape (m, num_classes), where each row contains the predicted logits
        for each class for a DNA sample. The model aggregates information across chunks
        per sample to produce a single prediction vector per sample.
    '''
    def __init__(self, dna_bert_model, num_classes, hidden_layer_sizes=[], cls_token_id=1, dropout=0.5):
        super(DNAOnlyClassifier, self).__init__()
        #Load the custom DNA BERT Model
        self.bert = DNABertModel(dna_bert_model, cls_token_id=cls_token_id)

        # Setup initial input size
        input_size = dna_bert_model.config.hidden_size

        # Create hidden layers if inputted
        modules = []
        for hidden_layer_size in hidden_layer_sizes:
            modules.append(nn.Linear(input_size, hidden_layer_size))
            modules.append(nn.ReLU()) 
            modules.append(nn.Dropout(dropout))
            input_size = hidden_layer_size

        # Add final classification head and group modules together
        modules.append(nn.Linear(input_size, num_classes))
        self.classifier = nn.Sequential(*modules)

        #softmax the output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, DNA_chunk_embeddings, DNA_attention_mask, return_probs=True):
        #Process through the custom BERT model
        dna_bert_outputs = self.bert(DNA_chunk_embeddings, attention_mask=DNA_attention_mask)

        # Extract the embeddings of the CLS token and run through classifier
        dna_embeddings = dna_bert_outputs[:, 0, :] # Take the CLS token from custom DNABERT model

        # Perform classification
        logits = self.classifier(dna_embeddings)

        #Determine if probabilities are a desired output
        if return_probs:
            probs = self.softmax(logits)
            return probs
        else:
            return logits

class ProteinOnlyClassifier(nn.Module):

    def __init__(self, protein_model, num_classes, hidden_layer_sizes=[], dropout=0.5, init_mode=''):
        super(ProteinOnlyClassifier, self).__init__()
        
        # Setup initial input size
        input_size = protein_model.config.hidden_size

        # Create hidden layers if inputted
        modules = []
        for hidden_layer_size in hidden_layer_sizes:
            layer = nn.Linear(input_size, hidden_layer_size)
            if init_mode == 'kaiming_normal':
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif init_mode == 'kaiming_uniform':
                init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
            modules.append(layer)
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout)) 
            input_size = hidden_layer_size

        # Add final classification head and group modules together
        modules.append(nn.Linear(input_size, num_classes))
        self.classifier = nn.Sequential(*modules)

        #softmax the output
        self.softmax = nn.Softmax(dim=1)


    def forward(self, protein_embeddings, return_probs=True):

        # Perform classification
        logits = self.classifier(protein_embeddings)

        #Determine if probabilities are a desired output
        if return_probs:
            probs = self.softmax(logits)
            return probs
        else:
            return logits
        

class DualHeadProteinOnlyClassifier(nn.Module):

    def __init__(self, protein_model, num_main_classes, hidden_layer_sizes=[], dropout=0.5, init_mode=''):
        super(DualHeadProteinOnlyClassifier, self).__init__()
        
        # Setup initial input size
        input_size = protein_model.config.hidden_size

        # Create hidden layers if inputted
        modules = []
        for hidden_layer_size in hidden_layer_sizes:
            layer = nn.Linear(input_size, hidden_layer_size)
            if init_mode == 'kaiming_normal':
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif init_mode == 'kaiming_uniform':
                init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
            modules.append(layer)
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout)) 
            input_size = hidden_layer_size

        # Shared layers
        self.shared_layers = nn.Sequential(*modules)

        # Binary classification head for detecting if it's in the main classes or "Other"
        self.binary_classifier = nn.Linear(input_size, 1)  # Binary classifier: Main classes or Other
        self.sigmoid = nn.Sigmoid()

        # Classifier for main classes
        self.main_class_classifier = nn.Linear(input_size, num_main_classes)  # Main classes classifier
        self.softmax = nn.Softmax(dim=1)

    def forward(self, protein_embeddings, return_probs=True):

        shared_output = self.shared_layers(protein_embeddings)

        # Perform binary classification on "Other" or "Not Other"
        binary_logits = self.binary_classifier(shared_output)
        binary_probs = self.sigmoid(binary_logits)

        # Main class classification
        main_class_logits = self.main_class_classifier(shared_output)
        main_class_probs = self.softmax(main_class_logits)  

        return binary_probs, main_class_probs