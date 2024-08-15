import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import h5py


class ProteinDataset(Dataset):
    '''
    A custom dataset class for handling and preprocessing protein sequences associated with phages,
    tailored for input into a BERT-based model. This class groups protein data by phage identifier
    (refseq_id), tokenizes the sequences, pads them to a fixed length, and ensures a fixed number
    of protein sequences per batch.

    Attributes:
        df_protein_labeled (DataFrame): A pandas DataFrame containing labeled protein data with columns
                                        for 'refseq_id', 'protein_sequence', and 'host_name'.
        tokenizer (Tokenizer): An instance of a tokenizer compatible with BERT models, used for converting
                               protein sequences into token ids.
        context_length (int): The fixed length of the downstream BERT model to which all protein sequences
                              will be padded or truncated.
        max_proteins (int): The fixed number of protein sequences each phage should have in the dataset.

    Methods:
        __len__(): Returns the number of unique phages in the dataset.
        __getitem__(idx): Retrieves a single processed item from the dataset by index. This item includes:
                          - 'refseq_id': The reference sequence ID of the phage.
                          - 'protein_sequences': A tensor of token ids for the protein sequences, padded to
                                                 ensure uniform length and number of sequences.
                          - 'attention_mask': A binary tensor indicating which tokens are padding and which
                                              are actual data.
                          - 'host_name': The name of the host associated with the phage.  Equivalent to the
                                         label.

    '''


    def __init__(self, df_protein_labeled, tokenizer, device, context_length=1024, max_proteins=10):
        #Save the basic information
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.max_proteins=10
        self.device = device

        # Select columns required for protein information
        self.protein_df = df_protein_labeled[['refseq_id', 'protein_sequence', 'host_name']]

    def __len__(self):
        return len(self.protein_df)

    def __getitem__(self, idx):
        data_row = self.protein_df.iloc[idx]
        protein_sequences = data_row['protein_sequence']
        refseq_id = data_row['refseq_id']
        host_name = data_row['host_name'] # Assume all host names are the same for a given phage

        # Tokenize protein sequences for appropriate input to BERT model
        tokenized_sequences = [self.tokenizer.encode(seq, add_special_tokens=True) for seq in protein_sequences]

        # Convert to tensor and pad extra space
        # Note: Each phage will have a different max sized protein.  Will deal with this by padding down below
        padded_sequences = pad_sequence([torch.tensor(seq) for seq in tokenized_sequences], batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Add additional padding to make all tokenzied vectors equivalent to context_length in length
        padded_sequences = torch.nn.functional.pad(
                          padded_sequences, (0, self.context_length - padded_sequences.shape[1]),
                          mode='constant', value=self.tokenizer.pad_token_id)

        # Pad the number of sequences so there are always max_proteins number of proteins per phage
        if len(padded_sequences) < self.max_proteins:
            # Calculate how many dummy sequences are needed
            num_dummy_sequences = self.max_proteins - len(padded_sequences)
            dummy_sequences = torch.full((num_dummy_sequences, self.context_length), self.tokenizer.pad_token_id)
            padded_sequences = torch.cat([padded_sequences, dummy_sequences], dim=0)

        # Create an attention mask for the sequences
        # Attention mask has the same shape as the padded_sequences tensor, and contains 1 where the tensor is not padding
        attention_mask = (padded_sequences != self.tokenizer.pad_token_id).long()

        # Create a downstream attention mask that denotes if a protein is present or if it's a padding protein
        downstream_attention_mask = attention_mask.squeeze()[:,0]

        return {
            'refseq_id': refseq_id,
            'protein_sequences': padded_sequences.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'downstream_attention_mask': downstream_attention_mask.to(self.device),
            'host_name': host_name
        }

class DNADataset(Dataset):
    '''
    A custom dataset class for handling and preprocessing DNA sequences associated with phages,
    tailored for input into a BERT-based model. This class groups DNA data by phage identifier
    (refseq_id), chunks up the DNA sequences into 100 chunks of 512 tokens, padding them to a
    fixed length to ensure a fixed number of chunks per phage.

    Attributes:
        df_DNA_labeled (DataFrame): A pandas DataFrame containing labeled protein data with columns
                                        for 'refseq_id', 'gene_sequence', and 'host_name'.
        tokenizer (Tokenizer): An instance of a tokenizer compatible with DNABERT models, used for converting
                                DNA sequences into token ids.
        overlap_token_count (int): The number of tokens of overlap between to adjacent chunks

    Methods:
        __len__(): Returns the number of unique phages in the dataset.
        __getitem__(idx): Retrieves a single processed item from the dataset by index. This item includes:
                            - 'refseq_id': The reference sequence ID of the phage.
                            - 'gene_sequence': A tensor of token ids for chunked DNA sequences, padded to a
                                                dimension of (nxc) where n is number of chunks and c is the
                                                 context length of the DNABERT model
                            - 'attention_mask': A binary tensor indicating which tokens are padding and which
                                                are actual data.
                            - 'downstream_attention_mask': A binary tensory indicating which chunks are entirely
                                                           padded chunks and should not be considered during inference
                            - 'host_name': The name of the host associated with the phage.  Equivalent to the
                                            label.

    '''

    def __init__(self, df_DNA_labeled, tokenizer, device, overlap_token_count=50):
        #Save the basic information
        self.tokenizer = tokenizer
        self.overlap_token_count = overlap_token_count
        self.device = device

        # Select columns required for DNA information
        self.DNA_df = df_DNA_labeled[['refseq_id', 'gene_sequence', 'host_name']]

    def __len__(self):
        return len(self.DNA_df)

    def __getitem__(self, idx):
        data_row = self.DNA_df.iloc[idx]
        gene_sequence = data_row['gene_sequence']
        refseq_id = data_row['refseq_id']
        host_name = data_row['host_name'] # Assume all host names are the same for a given phage

        padded_chunks, attention_mask, downstream_attention_mask = self.split_dna_sequence_with_overlap(gene_sequence,
                                                                                tokenizer = self.tokenizer,
                                                                                overlap_token_count=self.overlap_token_count,
                                                                                cls_token_id=self.tokenizer.cls_token_id,
                                                                                sep_token_id=self.tokenizer.sep_token_id,
                                                                                pad_token_id=self.tokenizer.pad_token_id)

        return {
            'refseq_id': refseq_id,
            'gene_sequence': padded_chunks.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'downstream_attention_mask': downstream_attention_mask.to(self.device),
            'host_name': host_name
        }

    @staticmethod
    def split_dna_sequence_with_overlap(dna_sequence, tokenizer, overlap_token_count=50, chunk_length=512,
                                        num_chunks=100, cls_token_id=1, sep_token_id=2, pad_token_id=3):
        """
        Splits a long DNA sequence into chunks of specified chunk_length, each chunk starting with a CLS token
        and ending with a SEP token.  Will overlap the chunks with overlap_token_count number of similar tokens.
        Will pad up to num_chunks with chunks consisting of only padding tokens but still starting with
        CLS and ending with SEP.

        Args:
        dna_sequence (str): The DNA sequence to be tokenized and split.
        chunk_length (int): Length of each sequence chunk, including CLS and SEP tokens.
        num_chunks (int): The desired number of chunks for input to DNABERT-2 model
        cls_token_id (int): Token ID for CLS.
        sep_token_id (int): Token ID for SEP.
        pad_token_id (int): Token ID for PAD.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing three tensors:
            - The first tensor (chunks) has shape (m, chunk_length), where m is the number of num_chunks.
            - The second tensor (attention_mask) has shape (m, chunk_length), indicating valid tokens (1) and padding (0).
            - The third tensor (downstream_attention_mask) has shape (m) where m is num_chunks. This will be used as input
            to the next DNABERT model to identify which chunks contain only padding tokens
        """
        chunks=[]
        attention_masks = []
        downstream_attention_masks = []

        #Tokenize the input dna
        inputs = tokenizer.encode(dna_sequence, add_special_tokens=False)

        #Break apart token list into chunks of size chunk_length
        adjusted_chunk_length = chunk_length - 2
        for i in range(0, len(inputs), (adjusted_chunk_length - overlap_token_count)):
            chunk = inputs[i:i+adjusted_chunk_length]
            # Add CLS and SEP tokens
            chunk = [cls_token_id] + chunk + [sep_token_id]
            chunks.append(torch.tensor(chunk, dtype=torch.long))
            # Create attention mask for this chunk
            mask = [1] * len(chunk)
            attention_masks.append(torch.tensor(mask, dtype=torch.long))
            downstream_attention_masks.append(torch.tensor([1], dtype=torch.long))

        # Pad the sequence of chunks and masks if necessary
        if len(chunks) < num_chunks:
            num_pad_chunk_to_add = num_chunks - len(chunks)
            # Each padding chunk is just a chunk of pad_token_id
            padding_chunk = [cls_token_id] + [pad_token_id]*adjusted_chunk_length + [sep_token_id]
            padding_mask = [0] * chunk_length
            for _ in range(num_pad_chunk_to_add):
                chunks.append(torch.tensor(padding_chunk, dtype=torch.long))
                attention_masks.append(torch.tensor(padding_mask, dtype=torch.long))
                downstream_attention_masks.append(torch.tensor([0], dtype=torch.long))

        #Convert all lists of tensors into tensors
        padded_chunks = pad_sequence(chunks, batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        downstream_attention_mask = pad_sequence(downstream_attention_masks, batch_first=True, padding_value=0).squeeze()

        return padded_chunks, attention_mask, downstream_attention_mask

class CLSEmbeddingsDataset(Dataset):
    '''
    A custom PyTorch Dataset class that loads CLS embeddings from an HDF5 file for specific reference sequences (refseq_id).

    This dataset is designed to handle the loading of precomputed embeddings and their corresponding attention masks
    stored in an HDF5 file. Each entry in the dataset corresponds to one refseq_id, providing the embeddings and
    attention masks as tensors.

    Attributes:
        hdf5_file (str): Path to the HDF5 file containing the embeddings and attention masks.
        refseq_ids (list): A list of refseq_ids for which embeddings and masks need to be loaded.
        data (h5py.File): An open HDF5 file object from which data is read.

    Methods:
        __len__():
            Returns the number of refseq_ids, indicating the number of items in the dataset.

        __getitem__(idx):
            Retrieves the embeddings and attention mask for the refseq_id at the specified index.
            Returns a dictionary containing:
                'chunk_embeddings' (Tensor): The chunk CLS embeddings for the specified refseq_id.
                'attention_mask' (Tensor): The attention mask corresponding to the embeddings.
                '''

    def __init__(self, hdf5_file, refseq_ids, labels):
        self.file = hdf5_file
        self.refseq_ids = refseq_ids
        self.labels = labels

    def __len__(self):
        return len(self.refseq_ids)

    def __getitem__(self, idx):

        with h5py.File(self.file, 'r') as data:
            refseq_id = self.refseq_ids[idx]
            label = self.labels[idx]
            chunk_embeddings = torch.tensor(np.array(data[refseq_id]['embeddings']), dtype=torch.float) # Load embeddings
            attention_mask = torch.tensor(np.array(data[refseq_id]['attention_mask']), dtype=torch.long) # Load attention mask

            # Adjust for similarly labeled hosts
            if label == 'Mycolicibacterium smegmatis':
                label = 'Mycolicibacterium smegmatis MC2 155'
            if label == 'Escherichia coli K-12':
                label = 'Escherichia coli'
            if label == 'Salmonella enterica subsp. enterica serovar Typhimurium':
                label = 'Salmonella enterica'
            if label == 'Salmonella enterica subsp. enterica serovar Enteritidis':
                label = 'Salmonella enterica'
            if label == 'Escherichia':
                label = 'Escherichia coli'

            return {'refseq_id': refseq_id, 'chunk_embeddings': chunk_embeddings,
                    'attention_mask': attention_mask, 'label': label}

    def close(self):
        self.data.close()

class ProteinAverageEmbeddingsDataset(CLSEmbeddingsDataset):
    ''' Child class of the CLSEmbeddingsDataset class, used to input the CLS
    protein embeddings for a given phage and output the average values across
    all proteins provided for the phage.  NaN values, or padded proteins, are
    ignored'''
    def __getitem__(self, idx):
        # Retrieve data from parent class
        item = super().__getitem__(idx)

        # Average the values across all proteins in a sample
        # Invalid values are nan and will not be included in the calculation
        averaged_embeddings = item['chunk_embeddings'].nanmean(dim=0) 

        return {'refseq_id': item['refseq_id'], 'embeddings': averaged_embeddings, 'label': item['label']}

class ProteinMaxEmbeddingsDataset(CLSEmbeddingsDataset):
    ''' Child class of the CLSEmbeddingsDataset class, used to input the CLS
    protein embeddings for a given phage and output the maximum absolute values 
    across all proteins provided for the phage.  NaN values, or padded proteins, 
    are ignored'''
    def __getitem__(self, idx):
        # Retrieve data from parent class
        item = super().__getitem__(idx)
        embeddings = item['chunk_embeddings']

        # Replace nan values with 0 to allow for max abs value calculation
        valid_tensor = torch.where(torch.isnan(embeddings), torch.full_like(embeddings, float(0)), embeddings)

        # Convert tensor to absolute value to find the most extreme values
        abs_tensor = torch.abs(valid_tensor)

        # Find indexes of most extreme values
        max_values, max_indices = abs_tensor.max(dim=0)

        # Keep most extreme values from original matrix based on index location
        max_embeddings = valid_tensor.gather(0, max_indices.unsqueeze(0)).squeeze()

        # # Take max value across all proteins in a sample
        # max_embeddings = valid_tensor.max(dim=0) 

        return {'refseq_id': item['refseq_id'], 'embeddings': max_embeddings, 'label': item['label']}
    
class CombinedDataset(Dataset):
    '''
    A custom PyTorch Dataset class that loads and combines data from both
    DNA embeddings and protein embeddings datasets.

    Attributes:
        dna_dataset (Dataset): An instance of CLSEmbeddingsDataset for DNA embeddings.
        protein_dataset (Dataset): An instance of ProteinAverageEmbeddingsDataset or 
                                ProteinMaxEmbeddingsDataset for protein embeddings.
    '''

    def __init__(self, dna_dataset, protein_dataset):
        assert len(dna_dataset) == len(protein_dataset), "DNA and Protein datasets must be of the same length"
        self.dna_dataset = dna_dataset
        self.protein_dataset = protein_dataset

    def __len__(self):
        return len(self.dna_dataset)  

    def __getitem__(self, idx):
        dna_item = self.dna_dataset[idx]
        protein_item = self.protein_dataset[idx]
        
        # Combine data from both datasets into a single dictionary
        combined_item = {
            'refseq_id': dna_item['refseq_id'], 
            'dna_chunk_embeddings': dna_item['chunk_embeddings'],
            'dna_attention_mask': dna_item['attention_mask'],
            'protein_embeddings': protein_item['embeddings'],
            'label': dna_item['label'] 
        }
        return combined_item