import pickle
import torch
import json
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers import AutoTokenizer
from transformers import Pipeline

from load_model import load_model


class CustomPipeline(Pipeline):
    ''' """
    A custom pipeline for phage-bacteria interaction prediction, built to handle 
    protein sequences using a pre-trained Fusion model. This pipeline manages 
    preprocessing, model inference, and postprocessing, with support for multi-class 
    and binary classification outputs.

    Parameters:
    -----------
    model : torch.nn.Module
        The pre-trained model used for predictions.

    tokenizer : AutoTokenizer
        Tokenizer for the protein model, used to encode sequences. Uses ESM-2 650M
        model.

    label_classes : list of str
        List of class labels for multi-class predictions.

    device : str
        Device on which the model is loaded ('cpu' or 'cuda').

    Methods:
    --------
    from_pretrained(model_name_or_path, device='cpu') -> CustomPipeline:
        Class method to load the pipeline from a pre-trained model directory.

    preprocess(inputs) -> dict:
        Tokenizes and prepares protein sequences for model input. Returns tokenized 
        sequences and attention masks.

    _forward(model_inputs) -> dict:
        Runs the preprocessed inputs through the model to obtain classification outputs.

    postprocess(model_outputs) -> list of tuple:
        Processes model outputs into human-readable predictions. Returns a list of tuples 
        containing the predicted class label and confidence score.

    Attributes:
    -----------
    model : torch.nn.Module
        The pre-trained model for classification.

    tokenizer : AutoTokenizer
        Tokenizer for encoding protein sequences.

    device : str
        The device used for model inference.

    label_classes : list of str
        Class labels for multi-class classification.'''

    def __init__(self, model, tokenizer, label_classes, device):
        """
        Initialize the pipeline.

        Args:
            model (torch.nn.Module): The model to use in the pipeline.
            device (str): Device on which the model is loaded.
        """
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.label_classes = label_classes

    @classmethod
    def from_pretrained(cls, model_name_or_path, device='cpu'):
        """
        Load the pipeline from a pretrained model.

        Args:
            model_name_or_path (str): Path to the model directory or pretrained model name.
            device (str): Device to load the model on ('cpu' or 'cuda').

        Returns:
            CustomPipeline: An instance of the CustomPipeline class.
        """
        # Load configuration
        with open(f'{model_name_or_path}/config.json', 'r') as f:
            config = json.load(f)

        # Load model
        model = load_model(model_name_or_path, config['model_params'], device=device)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model_params']['protein_model_name'])

        # Load label encoder classes from .pkl file
        with open(f'{model_name_or_path}/label_encoder_classes.pkl', 'rb') as f:
            label_classes = pickle.load(f)

        # Return pipeline instance
        return cls(tokenizer=tokenizer, model=model, device=device, label_classes=label_classes)

    def _sanitize_parameters(self, **kwargs):
    # Define or adjust parameters (e.g., options for preprocessing, etc.)
        return {}, {}, {}

    def preprocess(self, inputs):
      """
      Tokenizes and prepares protein sequences, returning per-phage tokenized sequences and attention masks.

      Args:
          batch_phages (list of lists): Each inner list contains protein sequences for one phage.

      Returns:
          per_phage_tokenized_sequences (list of torch.Tensor): Tokenized protein sequences per phage.
          per_phage_attention_masks (list of torch.Tensor): Attention masks per phage.
      """
      batch_phages = inputs.get('protein_sequences')

      # Check if the input is a single list of proteins
      if isinstance(batch_phages, list) and all(isinstance(item, str) for item in batch_phages):
      # Wrap the single list into a list of lists
        batch_phages = [batch_phages]

      max_proteins = self.model.max_proteins

      per_phage_tokenized_sequences = []
      per_phage_attention_masks = []

      for i,phage_proteins in enumerate(batch_phages):
          # Truncate to max_proteins
          if len(phage_proteins) > max_proteins:
              print(f"Phage {i+1} contains {len(phage_proteins)} proteins, exceeding the limit of {max_proteins}.")
              print(f"Only the first {max_proteins} proteins will be used.")
              phage_proteins = phage_proteins[:max_proteins]

          # Tokenize proteins
          tokenized_sequences = [
              self.tokenizer.encode(seq, add_special_tokens=True) for seq in phage_proteins
          ]

          # Convert to tensors and pad each sequence
          padded_sequences = pad_sequence(
              [torch.tensor(seq) for seq in tokenized_sequences],
              batch_first=True,
              padding_value=self.tokenizer.pad_token_id
          ).to(self.device)

          # Create an attention mask
          attention_mask = (padded_sequences != self.tokenizer.pad_token_id).long()

          per_phage_tokenized_sequences.append(padded_sequences)
          per_phage_attention_masks.append(attention_mask)

      return {
            "input_tokens": per_phage_tokenized_sequences,
            "attention_masks": per_phage_attention_masks,
          }

    def _forward(self, model_inputs):
        '''Passes the preprocessed inputs through the model for inference.

        Args:
        -----
        model_inputs : dict
            A dictionary containing preprocessed inputs, specifically:
            - `'input_tokens'`: List of tokenized protein sequences as tensors.
            - `'attention_masks'`: List of attention masks corresponding to the input tokens.

        Returns:
        --------
        dict:
            A dictionary containing:
            - `'binary_probs'` (torch.Tensor): Binary classification probabilities indicating 
              whether each sample belongs to a main class or "Other".
            - `'main_class_probs'` (torch.Tensor): Multi-class classification probabilities 
              over the main class labels.'''
        
        return self.model(
            per_phage_tokenized_sequences=model_inputs['input_tokens'],
            per_phage_attention_masks=model_inputs['attention_masks'],
            return_probs=True
        )

    def postprocess(self, model_outputs):
        binary_probs = model_outputs['binary_probs']
        main_class_probs = model_outputs['main_class_probs']

        predictions = []
        for binary_prob, class_probs in zip(binary_probs, main_class_probs):
            # If binary classification predicts "Other"
            if binary_prob.item() < 0.5:  # Threshold for "Other"
                predictions.append(('Other', round(1-binary_prob.item(),4)))
            else:
                # Get the top class index and map to label
                top_class_idx = class_probs.argmax().item()
                top_class_label = self.label_classes[top_class_idx]
                predictions.append((top_class_label, round(class_probs[top_class_idx].item(),4)))

        return predictions
