from transformers import AutoModel, AutoTokenizer

def load_DNA_model(location='local', device=''):
    if location == 'colab':
        model_path = "/content/drive/MyDrive/DS_266_Final_Project_PBI/DNABERT-2/DNABERT-2-117M"
        DNA_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        DNA_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

    elif location == 'local':
        DNA_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)

    return DNA_model, DNA_tokenizer