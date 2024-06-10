import torch
import torch.nn as nn

X = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                   [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1]],
                  [[1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
                   [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3]],
                  [[1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4],
                   [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]],
                  [[1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6],
                   [1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7]]], dtype=torch.float32)

multihead_attn = nn.MultiheadAttention(embed_dim=8, num_heads=2)

attn_output, attn_output_weights = multihead_attn(X, X, X)

print(attn_output.shape)
print(attn_output_weights.shape)