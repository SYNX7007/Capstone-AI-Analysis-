from torch.utils.data import DataLoader, random_split
from tokenizer import SimpleTokenizer
from dataset import DialogueDataset
from model import DialogueLLM
from utils import train_model, generate_response
from dialogues import get_all_dialogues
import torch
import json
import os
from datetime import datetime

def save_model_checkpoint(model, tokenizer, save_dir, epoch, loss):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'vocab': {
            'word_to_id': tokenizer.char_to_idx,  # Changed from word_to_id
            'id_to_word': tokenizer.idx_to_char,  # Changed from id_to_word
            'vocab_size': tokenizer.vocab_size
        }
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_model_checkpoint(checkpoint_path, model, tokenizer):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore tokenizer state
    tokenizer.word_to_id = checkpoint['vocab']['word_to_id']
    tokenizer.id_to_word = checkpoint['vocab']['id_to_word']
    tokenizer.vocab_size = checkpoint['vocab']['vocab_size']
    
    return checkpoint['epoch'], checkpoint['loss']

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            outputs = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Configuration
    config = {
        'batch_size': 32,
        'max_length': 128,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'validation_split': 0.1,
        'checkpoint_dir': './checkpoints',
        'save_every': 2  # Save checkpoint every N epochs
    }
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Load dialogues
    print("Loading dialogues...")
    dialogues = get_all_dialogues()
    
    # Initialize and fit tokenizer
    print("Initializing tokenizer...")
    tokenizer = SimpleTokenizer()
    all_texts = []
    for dialogue in dialogues:
        all_texts.append(dialogue['input'])
        all_texts.append(dialogue['response'])
    tokenizer.fit(all_texts)
    
    # Create dataset
    print("Creating dataset...")
    full_dataset = DialogueDataset(
        dialogues=dialogues,
        tokenizer=tokenizer,
        max_length=config['max_length']
    )
    
    # Split into train and validation sets
    dataset_size = len(full_dataset)
    val_size = int(config['validation_split'] * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    print("Initializing model...")
    model = DialogueLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_length']
    )
    
    # Training loop with validation and checkpointing
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Train for one epoch
        train_loss = train_model(
            model=model,
            dataloader=train_dataloader,
            num_epochs=1,
            device=DEVICE
        )
        
        # Evaluate on validation set
        val_loss = evaluate_model(model, val_dataloader, DEVICE)
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_model_checkpoint(
                model, tokenizer, config['checkpoint_dir'], epoch + 1, val_loss
            )
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Regular checkpoint saving
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = save_model_checkpoint(
                model, tokenizer, config['checkpoint_dir'], epoch + 1, val_loss
            )
            print(f"Saved regular checkpoint to {checkpoint_path}")
    
    print("Training completed!")
    
    # Test generation with multiple samples
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        input_ids = torch.tensor([tokenizer.encode(user_input)])
        response = generate_response(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            max_length=config['max_length'],
            temperature=0.7,
            device=DEVICE
        )
        print("NPC:", response)
    
    print("\nTesting generation with multiple samples:")
    model.eval()
    for test_input in user_input:
        input_ids = torch.tensor([tokenizer.encode(test_input)])
        response = generate_response(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            max_length=config['max_length'],
            temperature=0.7,
            device=DEVICE
        )
        print(f"\nInput: {test_input}")
        print(f"Generated response: {response}")

if __name__ == "__main__":
    main()

# tokenizer.py - Tokenizer implementation
class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        # Build vocabulary from texts
        unique_words = set()
        for text in texts:
            words = text.split()
            unique_words.update(words)
        
        # Create mappings
        self.word_to_id = {word: idx for idx, word in enumerate(sorted(unique_words))}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text):
        return [self.word_to_id.get(word, self.word_to_id['<unk>']) 
                for word in text.split()]
    
    def decode(self, ids):
        return ' '.join([self.id_to_word[id] for id in ids])

# dataset.py - Dataset handling
import torch
from torch.utils.data import Dataset

class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for dialogue in dialogues:
            input_ids = tokenizer.encode(dialogue['input'])
            target_ids = tokenizer.encode(dialogue['response'])
            
            # Pad or truncate sequences
            input_ids = self._pad_sequence(input_ids)
            target_ids = self._pad_sequence(target_ids)
            
            self.examples.append({
                'input_ids': torch.tensor(input_ids),
                'target_ids': torch.tensor(target_ids)
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def _pad_sequence(self, seq):
        if len(seq) > self.max_length:
            return seq[:self.max_length]
        return seq + [0] * (self.max_length - len(seq))

# model.py - Model architecture
import torch
import torch.nn as nn

class DialogueLLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff
            ),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output_layer(x)

import torch
import torch.nn as nn
import math
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensure the positional encoding matches the input tensor's sequence length
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :x.size(-1)]

# utils.py - Training and generation utilities
import torch
import torch.nn.functional as F

def train_model(model, dataloader, num_epochs, device='cuda'):
    # Move the model to the specified device
    model = model.to(device)
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            # Move the batch tensors to the same device as the model
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch['input_ids'])
            
            # Compute the loss
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch['target_ids'].view(-1)
            )
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

def generate_response(model, input_ids, tokenizer, max_length=128, temperature=0.7, top_p=0.9, device='cuda'):
    # Move the model to the specified device
    model = model.to(device)
    
    model.eval()
    current_ids = input_ids.to(device)
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(current_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            next_token = sorted_indices[0, 0]
            if next_token == tokenizer.word_to_id['<eos>']:
                break
                
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return tokenizer.decode(current_ids[0].tolist())

# dialogues.py - Dialogue data management
def get_all_dialogues():
    # This could load from a database or file
    return [
        {
            'input': 'Hello, how are you?',
            'response': 'I am doing well, thank you for asking!'
        },
        # Add more dialogue examples here
    ]

def get_dialogue_set(set_name):
    # Could load specific dialogue sets (e.g., training, testing)
    pass