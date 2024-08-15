import torch  
from torch.utils.data import DataLoader  


def process_protein_to_CLS_embeddings(protein_dataloader, protein_model, device=''):
  '''
  Processes protein sequences from a DataLoader, extracts CLS embeddings using a ESM2
  model, and groups them by phage.

  This function takes the tensor of 10 proteins, each length 1024 tokenized inputs,
  and runs through an ESM2 model.  The CLS embeddings are obtained for each protein,
  to be stored in a database for future combination.  Can be combined with some method
  (mean, max, concat, etc.) based on user downstream decision.

  Stored as dictionaries of CLS embeddings and attention masks keyed by refseq_id '''

  embeddings_by_phage = {}
  attention_mask_by_phage = {}

  for j,data in enumerate(protein_dataloader):

      # Extract the tokenized chunks, downstream attention mask,
      # and refseq_id from dataloader
      sequences = data['protein_sequences'].squeeze().to(device)
      attention_mask = data['attention_mask'].squeeze().to(device)
      downstream_attention_mask = data['downstream_attention_mask'].squeeze().to(device)
      refseq_id = data['refseq_id'][0]

      # Run sample through protein model
      with torch.no_grad():
          output = protein_model(sequences, attention_mask=attention_mask)
          cls_output = output['last_hidden_state'][:,0,:]

      # Capture the data
      embeddings_by_phage[refseq_id] = cls_output
      attention_mask_by_phage[refseq_id] = downstream_attention_mask

        #Reduce memory imprint
      del cls_output
      torch.cuda.empty_cache()

      if (j+1) % 10 == 0:
        print(f'Completed phage number {j+1} out of {len(protein_dataloader)}')

  return embeddings_by_phage, attention_mask_by_phage



def process_DNA_to_CLS_embeddings(DNA_dataloader, DNA_model):
    '''
    Processes DNA sequences from a DataLoader, extracts CLS embeddings using a DNABERT
    model, and groups them by phage.

    This function takes the tensor of 100 chunks of tokenized input and sub-batches the chunks
    to run through a pre-trained DNABERT model.  The CLS embeddings are obtained, concatenated
    together for a given phage and stored along with their corresponding downstream attention
    masks as dictionaries keyed by their refseq_ids.
    '''

    embeddings_by_phage = {}
    attention_mask_by_phage = {}

    for j,data in enumerate(DNA_dataloader):

        # Extract the tokenized chunks, downstream attention mask,
        # and refseq_id from dataloader
        sequences = data['gene_sequence'].squeeze()
        attention_mask = data['attention_mask'].squeeze()
        downstream_attention_mask = data['downstream_attention_mask'].squeeze()
        refseq_id = data['refseq_id'][0]

        # Process in smaller sub-batches of 10
        sub_batch_size = 10
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(sequences), sub_batch_size):
                sub_batch_sequences = sequences[i:i + sub_batch_size]
                sub_batch_attention_mask = attention_mask[i:i + sub_batch_size]
                output = DNA_model(sub_batch_sequences, attention_mask=sub_batch_attention_mask)  # Process the sub-batch through DNABERT
                cls_output = output[0][:,0,:].detach()
                embeddings.append(cls_output)

        # Concatenate embeddings from subbatches into the same tensor
        concatenated_embeddings = torch.cat(embeddings, dim=0)

        # Add the embedding tensor and attention mask to dictionaries
        embeddings_by_phage[refseq_id] = concatenated_embeddings
        attention_mask_by_phage[refseq_id] = downstream_attention_mask

        del embeddings

        print(f'Completed phage number {j}')

    return embeddings_by_phage, attention_mask_by_phage