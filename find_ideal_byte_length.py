import sentencepiece as spm
import torch

# Rows for `nn.Embedding` in DictionaryFactory / `build_byte_table`:
# - 0: padding (Embedding.padding_idx)
# - 1–256: UTF-8 byte b stored as (b + 1)
# - 257 + meta_id: SentencePiece control/unknown token ids (e.g. <pad>=0 … <unk>=3 → 257..260)
# Max index is 260 ⇒ num_embeddings must be **261**.
BYTE_BARCODE_VOCAB_SIZE = 261


def analyze_vocab_lengths(spm_model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)
    
    vocab_size = sp.get_piece_size()
    
    max_len = 0
    longest_token = ""
    
    # Store prefixes to check for collisions
    tokens_by_bytes = []

    for id in range(vocab_size):
        piece = sp.id_to_piece(id)
        
        # Skip control tokens like <unk> or <s>
        if sp.is_control(id) or sp.is_unknown(id):
            continue
            
        # In the analyze script (and your final tensor building script):
        if piece.startswith('<0x') and piece.endswith('>'):
            # It's a special fallback. Get the hex, and SHIFT IT by +256
            byte_val = int(piece[3:-1], 16)
            byte_vals = [byte_val + 256] 
        else:
            # It's a standard text string. 
            # (Remember we shift by +1 to reserve 0 for padding)
            byte_vals = [b + 1 for b in piece.encode('utf-8')]        

        byte_len = len(byte_vals)
        tokens_by_bytes.append((piece, byte_vals))
        
        if byte_len > max_len:
            max_len = byte_len
            longest_token = piece

    print(f"--- VOCABULARY ANALYSIS ---")
    print(f"Absolute maximum byte length: {max_len}")
    print(f"The longest token is: '{longest_token}'\n")

    # Let's find the mathematically ideal cutoff to avoid collisions
    for cutoff in range(1, max_len + 2):
        seen_prefixes = set()
        collisions = 0
        
        for piece, byte_vals in tokens_by_bytes:
            # Get the truncated byte prefix
            prefix = tuple(byte_vals[:cutoff])
            
            if prefix in seen_prefixes:
                collisions += 1
            else:
                seen_prefixes.add(prefix)
                
        if collisions == 0:
            print(f"✅ Ideal `max_bytes`: {cutoff} (0 collisions)")
            return cutoff
            break
        else:
            print(f"At cutoff={cutoff}, there are {collisions} collisions.")
    return None        

max_bytes = analyze_vocab_lengths('./data/tokenizers/fineweb_1024_bpe.model')


def build_byte_table(spm_model_path, max_bytes):
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)

    # get the vocab size
    vocab_size = sp.get_piece_size()

    # initialize the cheat table
    cheat_table = torch.zeros((vocab_size, max_bytes), dtype=torch.long)

    for token_id in range(vocab_size):
        token = sp.id_to_piece(token_id)

        # RULE 1: Control / unknown tokens → rows 257 + token_id (needs BYTE_BARCODE_VOCAB_SIZE).
        if sp.is_control(token_id) or sp.is_unknown(token_id):
            print(f"Control token: {token} (token_id: {token_id})")
            idx = 257 + token_id
            if idx >= BYTE_BARCODE_VOCAB_SIZE:
                raise ValueError(
                    f"{token!r} token_id={token_id} → barcode row {idx}, "
                    f"but BYTE_BARCODE_VOCAB_SIZE={BYTE_BARCODE_VOCAB_SIZE}. "
                    "Increase BYTE_BARCODE_VOCAB_SIZE or remap controls."
                )
            cheat_table[token_id, 0] = idx
            continue
        
        # RULE 2: Handle Special Fallback Tokens (<0xEA>) and map them to
        if token.startswith('<0x') and token.endswith('>'):
            # It's a special fallback. Get the hex, and SHIFT IT by +1
            byte_val = int(token[3:-1], 16)
            cheat_table[token_id, 0] = byte_val + 1
            continue
        
        # RULE 3: Handle Standard Text Tokens
        # Decode the piece to get the actual string, then convert to UTF-8 bytes
        # (sp.decode_id removes the weird SentencePiece '_' spaces and gives pure text)

        raw_text = sp.decode([token_id])
        raw_bytes = list(raw_text.encode('utf-8'))

        # If the token is empty after decoding (happens with pure space tokens), 
        # we fall back to encoding the literal piece string.
        if len(raw_bytes) == 0:
            raw_bytes = list(token.encode('utf-8'))

        # Now we can safely convert to byte values and store them
        # Fill the slots up to max_bytes
        for i in range(min(len(raw_bytes), max_bytes)):
            # Standard shift: +1
            cheat_table[token_id, i] = raw_bytes[i] + 1

    return cheat_table

if __name__ == "__main__":
    max_bytes = analyze_vocab_lengths('./data/tokenizers/fineweb_1024_bpe.model')   
    my_byte_table = build_byte_table('./data/tokenizers/fineweb_1024_bpe.model', max_bytes)

    # Save it to disk so the model can load it instantly during training
    torch.save(my_byte_table, 'byte_table.pt')
    print("Byte table generated and saved! Shape:", my_byte_table.shape)