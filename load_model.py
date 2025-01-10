import torch
from model import DualHeadProteinOnlyClassifier

def load_model(model_name_or_path, config, device):
    '''   
    Loads a pre-trained DualHeadProteinOnlyClassifier model with its saved weights.

    This function initializes the model using the provided configuration, 
    loads the saved weights for specific components, and prepares it for inference or fine-tuning.

    Args:
    -----
    model_name_or_path : str
        Path to the directory containing the saved model weights and configuration.

    config : dict
        Configuration dictionary containing model parameters, including:
        - `protein_model_name`: Name of the pre-trained protein model.
        - `num_main_classes`: Number of main output classes.
        - Additional parameters for initializing the model.

    device : str
        The device on which to load the model ('cpu' or 'cuda').

    Returns:
    --------
    DualHeadProteinOnlyClassifier:
        An instance of the DualHeadProteinOnlyClassifier with loaded weights.
    '''
    # Add device to config file
    config['device'] = device

    # Instantiate model
    model = DualHeadProteinOnlyClassifier(**config).to(device)

    # Load saved model weights
    weights_path = f"{model_name_or_path}/pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location=device)

    # Only load the weights for the shared_layers, binary_classifier, and main_class_classifier,
    # as those were the only parts saved
    model.shared_layers.load_state_dict(
        {k[len("shared_layers."):]: v for k, v in state_dict.items() if k.startswith("shared_layers.")}
    )
    model.binary_classifier.load_state_dict(
        {k[len("binary_classifier."):]: v for k, v in state_dict.items() if k.startswith("binary_classifier.")}
    )
    model.main_class_classifier.load_state_dict(
        {k[len("main_class_classifier."):]: v for k, v in state_dict.items() if k.startswith("main_class_classifier.")}
    )

    return model