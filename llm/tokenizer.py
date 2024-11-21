class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {
            '<PAD>': 0, 
            '<START>': 1, 
            '<END>': 2,
            '<UNK>': 3
        }
        self.idx_to_char = {
            0: '<PAD>', 
            1: '<START>', 
            2: '<END>',
            3: '<UNK>'
        }
        self.vocab_size = 4
        
    def fit(self, texts):
        # Add special tokens for formatting
        special_tokens = [' ']  # Add space as a special token
        
        # First add special tokens
        for token in special_tokens:
            if token not in self.char_to_idx:
                self.char_to_idx[token] = self.vocab_size
                self.idx_to_char[self.vocab_size] = token
                self.vocab_size += 1
        
        # Then add character tokens
        for text in texts:
            for char in text:
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = self.vocab_size
                    self.idx_to_char[self.vocab_size] = char
                    self.vocab_size += 1
    
    def encode(self, text, max_length=None):
        # Convert text to indices
        encoded = [self.char_to_idx['<START>']]
        
        # Split text into tokens and handle special tokens
        words = text.split()
        for i, word in enumerate(words):
            if i > 0:  # Add space between words
                encoded.append(self.char_to_idx[' '])
            for char in word:
                if char in self.char_to_idx:  # Only encode known characters
                    encoded.append(self.char_to_idx[char])
                else:
                    encoded.append(self.char_to_idx['<UNK>'])  # Use UNK for unknown chars
        
        encoded.append(self.char_to_idx['<END>'])
        
        if max_length is not None and len(encoded) > max_length:
            encoded = encoded[:max_length]
            
        return encoded
    
    def decode(self, indices):
        text = []
        space_needed = False
        
        for idx in indices:
            if idx == self.char_to_idx['<END>']:
                break
            if idx in [self.char_to_idx['<PAD>'], self.char_to_idx['<START>']]:
                continue
                
            token = self.idx_to_char.get(idx, '<UNK>')
            
            if space_needed and token != ' ':
                text.append(' ')
            text.append(token)
            space_needed = (token != ' ')
                
        return ''.join(text).strip()

