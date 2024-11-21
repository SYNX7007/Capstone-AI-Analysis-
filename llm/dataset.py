import torch
from torch.utils.data import Dataset

class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=512):
        self.dialogues = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Format dialogues as input-response pairs
        for i in range(0, len(dialogues), 2):
            if i + 1 < len(dialogues):
                self.dialogues.append((dialogues[i], dialogues[i + 1]))
        
    def __len__(self):
        return len(self.dialogues)
        
    def __getitem__(self, idx):
        input_text, response_text = self.dialogues[idx]
        
        # Format as "USER: {input} NPC: {response}"
        full_text = f"USER: {input_text} NPC: {response_text}"
        
        # Get encoded sequence
        encoded = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        # Pad or truncate to max_length - 1 to leave room for shifting
        if len(encoded) > self.max_length - 1:
            encoded = encoded[:self.max_length - 1]
        else:
            encoded.extend([self.tokenizer.char_to_idx['<PAD>']] * (self.max_length - 1 - len(encoded)))
        
        # Create input and target sequences
        input_ids = torch.tensor(encoded)
        target_ids = torch.tensor(encoded[1:] + [self.tokenizer.char_to_idx['<PAD>']])
        
        return input_ids, target_ids
