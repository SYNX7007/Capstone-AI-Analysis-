import torch
import torch.nn.functional as F

def train_model(model, dataloader, num_epochs=1, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    total_loss = 0
    for epoch in range(num_epochs):
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

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

def generate_response(model, input_ids, tokenizer, max_length=128, temperature=0.7, top_p=0.9, device='cuda'):
    model = model.to(device)
    model.eval()
    current_ids = input_ids.to(device)

    print("Input IDs:", current_ids)
    
    for _ in range(max_length):
        with torch.no_grad():
            # Ensure input doesn't exceed max_length
            if current_ids.size(1) > max_length:
                current_ids = current_ids[:, -max_length:]
            
            outputs = model(current_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            next_token = sorted_indices[0, 0]
            if next_token == tokenizer.char_to_idx['<END>']:
                break
                
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return tokenizer.decode(current_ids[0].tolist())
