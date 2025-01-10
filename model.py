import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel, AutoTokenizer, PretrainedConfig

class DualHeadProteinOnlyClassifier(nn.Module):

    '''A dual-head classification model for protein sequence analysis, designed for phage-bacteria interaction prediction.
    This model uses a pre-trained protein model (e.g., ESM) for feature extraction and provides both binary 
    and multi-class classification outputs. 

    Parameters:
    -----------
    protein_model_name : str
        Name of the pre-trained protein model to use (e.g., ESM) from Hugging Face's Transformers library.

    num_main_classes : int
        The number of output classes for multi-class classification.

    hidden_layer_sizes : list of int, optional
        Sizes of additional fully connected layers between the feature extractor and classification heads. 

    dropout : float, optional
        Dropout probability to apply after each hidden layer (default=0.5).

    init_mode : str, optional
        Initialization mode for weights in hidden layers. Options: 'kaiming_normal', 'kaiming_uniform', or '' for default initialization.

    device : str, optional
        The device to run the model on ('cpu' or 'cuda', default='cpu').

    Attributes:
    -----------
    protein_model : AutoModel
        The pre-trained protein sequence model used for feature extraction.

    tokenizer : AutoTokenizer
        The tokenizer corresponding to the pre-trained protein model.

    shared_layers : nn.Sequential
        A sequence of hidden layers shared by both classification heads.

    binary_classifier : nn.Linear
        Binary classification head for determining if a sample belongs to the main classes or "Other".

    main_class_classifier : nn.Linear
        Multi-class classification head for classifying samples into the main classes.

    sigmoid : nn.Sigmoid
        Activation function for the binary classifier.

    softmax : nn.Softmax
        Activation function for the multi-class classifier.

    Methods:
    --------
    can_generate():
        A mock method for compatibility with pipelines. Always returns False.

    forward(per_phage_tokenized_sequences, per_phage_attention_masks, return_probs=True):
        Processes tokenized protein sequences and computes classification probabilities.

        Args:
        -----
        per_phage_tokenized_sequences : list of torch.Tensor
            Tokenized sequences of proteins for each phage in the batch.

        per_phage_attention_masks : list of torch.Tensor
            Attention masks corresponding to the tokenized sequences.

        return_probs : bool, optional
            If True, returns probabilities for the classification outputs; otherwise, returns raw logits (default=True).

        Returns:
        --------
        dict:
            A dictionary containing:
                - `'binary_probs'` (torch.Tensor): Probabilities from the binary classification head.
                - `'main_class_probs'` (torch.Tensor): Probabilities from the multi-class classification head.
                - `'embeddings'` (torch.Tensor): Mean protein sequence embeddings for each phage.'''

    def __init__(self, protein_model_name, num_main_classes, hidden_layer_sizes=[], dropout=0.5, init_mode='', device='cpu'):
        super(DualHeadProteinOnlyClassifier, self).__init__()

        config_dict = {
            "protein_model_name": protein_model_name,
            "num_main_classes": num_main_classes,
            "hidden_layer_sizes": hidden_layer_sizes,
            "dropout": dropout,
            "init_mode": init_mode,
            "task_specific_params": None
        }

        self.config = PretrainedConfig(**config_dict)

        # Load ESM model from huggingface
        self.protein_model = AutoModel.from_pretrained(protein_model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
        self.context_length = 1024
        self.max_proteins=10
        self.device = device

        # Setup initial input size
        input_size = self.protein_model.config.hidden_size

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

    def can_generate(self):
        return False

    def forward(self, per_phage_tokenized_sequences, per_phage_attention_masks, return_probs=True):

        # Compute mean embeddings for each phage
        mean_embeddings = []
        for tokenized_sequences, attention_mask in zip(per_phage_tokenized_sequences, per_phage_attention_masks):
            # Run each phage's proteins through the model
            output = self.protein_model(tokenized_sequences, attention_mask=attention_mask)
            cls_output = output['last_hidden_state'][:, 0, :]  # CLS token representation

            # Compute mean embedding for the phage
            mean_embedding = cls_output.nanmean(dim=0)
            mean_embeddings.append(mean_embedding)

        # Combine all phage embeddings into a batch tensor
        mean_embeddings = torch.stack(mean_embeddings)  # Shape: (batch_size, embedding_dim)

        #####################
        # Pass through linear layers
        shared_output = self.shared_layers(mean_embeddings)

        # Perform binary classification on "Other" or "Not Other"
        binary_logits = self.binary_classifier(shared_output)
        binary_probs_batch = self.sigmoid(binary_logits)

        # Main class classification
        main_class_logits = self.main_class_classifier(shared_output)
        main_class_probs_batch = self.softmax(main_class_logits)

        return {'binary_probs': binary_probs_batch,
                'main_class_probs': main_class_probs_batch
        }