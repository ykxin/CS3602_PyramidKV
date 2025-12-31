
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from pyramidkv.pyramidkv_utils import init_pyramidkv

def gptneox_attn_forward_PyramidKV(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    # Initialize PyramidKV
    init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)

    has_layer_past = layer_past is not None

    # Compute QKV
    qkv = self.query_key_value(hidden_states)
    
    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)    
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]
    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    # PyramidKV compression reduces key length, but position_ids remain large (absolute positions).
    # We must ensure rotary_emb returns cos/sin tables large enough to cover position_ids.
    max_pos_needed = position_ids.max().item() + 1
    current_seq_len = key.shape[-2]
    if has_layer_past:
        current_seq_len += layer_past[0].shape[-2]
        
    seq_len = max(max_pos_needed, current_seq_len)
    
    cos, sin = self.rotary_emb(value, seq_len=seq_len)
    query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)
    key = torch.cat((key, key_pass), dim=-1)

    # PyramidKV Logic
    # We need to compress the KV cache if we are in the prefill stage (long sequence)
    # The original logic in llama_model.py checks if key_states.shape[-2] == kv_seq_len (which implies full sequence)
    # Here, 'key' is the current step's key. 'layer_past' is the past.
    
    # If we are processing a prompt (layer_past is None), we can compress.
    # If we are decoding (layer_past is not None), we usually just append.
    
    # PyramidKV implementation in llama_model.py seems to rely on 'past_key_value' being a Cache object to handle updates.
    # But here we are dealing with tuples.
    
    # Let's adapt the logic:
    # If layer_past is None, we are processing the initial prompt. We apply PyramidKV compression.
    # If layer_past is Not None, we are generating tokens. We append to cache (no compression for new token usually, or maybe sliding window?)
    
    if not has_layer_past:
        # Initial prompt processing
        # Use PyramidKV to compress key/value
        # key: [batch, num_heads, seq_len, head_size]
        
        # num_key_value_groups is needed for update_kv. GPTNeoX usually has 1 group (MHA).
        num_key_value_groups = 1 # Assumption for Pythia/GPT-NeoX
        
        # We need to pass query, key, value to update_kv
        # update_kv expects: key_states, query_states, value_states, attention_mask, num_key_value_groups
        
        # Note: update_kv expects key_states in shape [bsz, num_heads, seq_len, head_dim] (based on transpose in llama_model)
        # Our key is already [batch, num_heads, seq_len, head_size]
        
        key_compress, value_compress = self.kv_cluster.update_kv(
            key, query, value, attention_mask, num_key_value_groups
        )
        
        # Debug Print (Only print once per layer or infrequently to avoid spam, but for benchmark once is fine)
        # if self.layer_idx == 0 and torch.distributed.is_initialized() == False:
        #     print(f"PyramidKV: Layer 0 compressed KV from {key.shape[-2]} to {key_compress.shape[-2]}")
        # elif self.layer_idx == 0:
        #     pass # Avoid distributed print spam
        
        # Use the compressed KV as the new 'present' (which will be layer_past for next step)
        present = (key_compress, value_compress) if use_cache else None
        
        # Use compressed KV for attention calculation? 
        # In llama_model.py, it calls past_key_value.update with compressed states.
        # But for the *current* attention calculation, it uses the FULL states (implied by return key_states, value_states in else branch of llama_model)
        # Wait, in llama_model.py:
        # if key_states.shape[-2] == kv_seq_len:
        #    key_states_compress, value_states_compress = self.kv_cluster.update_kv(...)
        #    past_key_value.update(key_states_compress, value_states_compress, ...)
        
        # It updates the cache with COMPRESSED states.
        # But what does it use for attention?
        # llama_model.py lines 164:
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        # Here 'key_states' is the one BEFORE compression? 
        # No, update_kv returns 'key_states, value_states'. 
        # In llama_model.py line 157, it assigns result to key_states_compress, value_states_compress.
        # It does NOT overwrite key_states used for attention calculation?
        # Wait, line 164 uses 'key_states'. Line 157 defines 'key_states_compress'.
        # So it seems for the CURRENT step (prefill), it uses FULL attention.
        # But it stores COMPRESSED KV for future steps.
        
        # So:
        # 1. Compute attention using FULL key, value.
        # 2. Store COMPRESSED key, value into 'present'.
        
        present = (key_compress, value_compress) if use_cache else None
        
        # Attention calculation uses uncompressed key/value for the prompt itself
        # This ensures high quality for the prompt processing, but reduces cache size for generation.
        
    else:
        # Decoding step
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None
        
    # Attention calculation
    
    # Fix attention_mask if shapes mismatch due to compression
    # The provided attention_mask assumes full sequence length, but our key is compressed.
    # Since compressed KV contains valid tokens, we can use a mask of zeros (all visible).
    if attention_mask is not None and key.shape[-2] != attention_mask.shape[-1]:
        # Create a new mask of zeros matching key length
        # shape: [batch, 1, 1, key_seq_len]
        attention_mask = torch.zeros(
            (attention_mask.shape[0], 1, 1, key.shape[-2]), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        
        # Debug Print for shapes
        # if self.layer_idx == 0:
        #      print(f"DEBUG: Layer 0 corrected mask shape: {attention_mask.shape}, key shape: {key.shape}")

    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
    
    # Merge heads
    if hasattr(self, "_merge_heads"):
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
    else:
        # Fallback if _merge_heads is missing (e.g. in some SDPA implementations or versions)
        # attn_output: [batch, heads, seq_len, head_dim]
        # target: [batch, seq_len, hidden_size]
        bsz, num_heads, seq_len, head_dim = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)
    
    # Project (Dense)
    attn_output = self.dense(attn_output)
    
    # Dropout
    attn_output = self.attention_dropout(attn_output)
    
    if output_attentions:
        return attn_output, present, attn_weights
    else:
        return attn_output, present
