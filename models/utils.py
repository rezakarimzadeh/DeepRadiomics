import torch.nn as nn


class MaskedBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x, pad_mask=None):
        """
        x: [B, T, F]
        pad_mask: [B, T] boolean, True for PAD positions, False for valid data
        """
        B, T, F = x.shape
        x_flat = x.view(B * T, F)

        # No mask -> just do normal BN
        if pad_mask is None:
            x_bn = self.bn(x_flat)
            return x_bn.view(B, T, F)

        # Flatten mask: True = pad, False = valid
        mask_flat = pad_mask.reshape(B * T)   # [B*T]
        valid_idx = ~mask_flat               # True where we have real data

        # Edge case: if a batch is entirely padding, just return x
        if valid_idx.sum() == 0:
            return x

        # Select only valid positions
        x_valid = x_flat[valid_idx]          # [N_valid, F]

        # Apply BN only on valid entries
        x_valid_bn = self.bn(x_valid)        # [N_valid, F]

        # Put them back into their original positions
        x_flat_out          = x_flat.clone()
        x_flat_out[valid_idx] = x_valid_bn

        x_out = x_flat_out.view(B, T, F)
        return x_out


class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(CustomBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x, pad_mask=None):
        # x: [B, T, F] -> reshape to [B*T, F]
        B, T, F = x.shape
        x_reshaped = x.view(B * T, F)
        x_bn = self.bn(x_reshaped)
        # reshape back to [B, T, F]
        x_out = x_bn.view(B, T, F)
        return x_out

