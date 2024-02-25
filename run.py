from transformers import TransformerForLM
from tokenizer import Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    "vocab_size": 1024,
    "context_size": 64,
    "hidden_size": 128,
    "num_layers": 8,
    "expand_size": 256,
    "num_heads": 8
}

pair2idx_path = "tokenizer\pair2idx.pkl"
idx2bytes_path = "tokenizer\idx2bytes.pkl"

path = "model\model00.pth"

model = TransformerForLM(**config)
model.load_state_dict(torch.load(path))

tokenizer = Tokenizer.from_pickle_files(pair2idx_path, idx2bytes_path)

def generate_token(text):
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor(tokens[-64:], device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(tokens_tensor)
        output = logits.argmax(dim=-1)[0, -1]

        idx = output.item()
        decoded = tokenizer.decode([idx])

    return decoded

def auto_regressive_generation(text, max_tokens):
    for i in range(max_tokens):
        decoded_token = generate_token(text[-256:])
        text += decoded_token
        yield decoded_token

text = '''BAPTISTA:
I follow you.

BIONDELLO:
Cambio!

LUCENTIO:
What sayest thou, Biondello?

BIONDELLO:
You saw my master wink and laugh upon you?

LUCENTIO:
Biondello, what of that?'''

if __name__ == "__main__":
    num_tokens = int(input())
    
    for token in auto_regressive_generation(text, num_tokens):
        print(token, end='')