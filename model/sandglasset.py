import torch.nn as nn
import torch
import numpy as np
from utils import overlap_and_add, reshape


class TasNetEncoder(nn.Module):
    def __init__(self, window_length=4, out_channels=256, out_features=128, kernel_size=1):
        super(TasNetEncoder, self).__init__()
        self.window_length = window_length
        self.ind = None

        self.conv = nn.Conv1d(
            in_channels=window_length,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
        )

        self.lin = nn.Linear(in_features=out_channels, out_features=out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input x: [batch_size, T]
        # after reshape x: [batch_size, M, L]
        x, self.ind, padding = reshape(x, self.window_length, ind=self.ind, dim=1)
        x_wave = x
        # after conv, relu x: [batch_size, E, L]
        x = self.conv(x)
        x = self.relu(x)
        # after permute x: [batch_size, L, E]
        batch_size, E, L = x.size()
        x = x.permute(0, 2, 1)
        # after lin x: [batch_size, L, D]
        x = self.lin(x)
        # after permute x: [batch_size, D, L]
        x = x.permute(0, 2, 1)

        return x, x_wave, padding


class SegmentationModule(nn.Module):
    def __init__(self, segment_length=256):
        super(SegmentationModule, self).__init__()
        self.segment_length = segment_length
        self.ind = None

    def forward(self, x):
        # input x: [batch_size, D, L]
        # after reshape x: [batch_size, D, K, S]
        x, self.ind, padding = reshape(x, self.segment_length, ind=self.ind, dim=2)
        return x, padding


class LocalSequenceProcessingRNN(nn.Module):
    def __init__(self, in_channels, hidden_size=128, num_layers=1):
        super(LocalSequenceProcessingRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )

        self.lin = nn.Linear(in_features=2 * hidden_size, out_features=in_channels, bias=True)
        self.ln = nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, x):
        batch_size, D, K, S = x.size()
        # after permute/reshape x: [batch_size * S, K, D]
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(batch_size * S, K, D)
        x_residual = x
        # after lstm x: [batch_size * S, K, 2 * H]
        x, _ = self.lstm(x)
        # after lin x: [batch_size * S, K, D]
        x = self.lin(x)
        x = self.ln(x) + x_residual
        # after reshape/permute x: [batch_size, D, K, S]
        x = x.reshape(batch_size, S, K, D)
        x = x.permute(0, 3, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, n_pos, dim=10000):
        super(PositionalEncoding, self).__init__()

        self.pe = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )

        self.pe[:, 0::2] = torch.FloatTensor(np.sin(self.pe[:, 0::2]))
        self.pe[:, 1::2] = torch.FloatTensor(np.cos(self.pe[:, 1::2]))
        self.pe = torch.Tensor(self.pe)
        self.pe.detach_()
        self.pe.requires_grad = False

    def forward(self, x):
        return self.pe[:, :x.size()[1]]


class SelfAttentiveNetwork(nn.Module):
    def __init__(self, in_channels, kernel_size, num_heads=8, dropout=0.1):
        super(SelfAttentiveNetwork, self).__init__()
        self.downsampling = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

        self.ln1 = nn.LayerNorm(normalized_shape=in_channels)
        self.san = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.pe = PositionalEncoding(in_channels)
        self.ln2 = nn.LayerNorm(normalized_shape=in_channels)

        self.upsampling = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x):
        batch_size, D, K, S = x.size()
        # after permute/reshape x: [batch_size * S, D, K]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size * S, D, K)
        # after downsampling x: [batch_size * S, D, floor(K / kernel_size)]
        x = self.downsampling(x)
        # after reshape/permute/reshape x: [batch_size * floor(K / kernel_size), S, D]
        x = x.reshape(batch_size, S, D, -1)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(-1, S, D)

        positional_encoding = torch.t(self.pe(x))
        positional_encoding = positional_encoding.unsqueeze(dim=0)
        x = self.ln1(x) + positional_encoding
        x_residual = x
        # after san x: [batch_size * floor(K / kernel_size), S, D]
        x, _ = self.san(x, x, x)
        x = self.ln2(x + x_residual)

        # after reshape/permute/reshape x: [batch_size * S, D, floor(K / kernel_size)]
        x = x.reshape(batch_size, -1, S, D)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size * S, D, -1)

        # after upsampling x: [batch_size * S, D, K'], where K' = floor(K / kernel_size) * kernel_size
        x = self.upsampling(x)
        # after reshape/permute x: [batch_size, D, K', S], where K' = floor(K / kernel_size) * kernel_size
        x = x.reshape(batch_size, S, D, -1)
        x = x.permute(0, 2, 3, 1)

        return x


class SandglassetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, hidden_size=128, num_layers=1, num_heads=8, dropout=0.1):
        super(SandglassetBlock, self).__init__()
        self.local_rnn = LocalSequenceProcessingRNN(
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.san = SelfAttentiveNetwork(
            in_channels,
            kernel_size,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.local_rnn(x)
        x = self.san(x)
        return x


class MaskEstimation(nn.Module):
    def __init__(self, in_channels, out_channels, source_num, encoded_frame_dim, window_length, kernel_size=1):
        super(MaskEstimation, self).__init__()
        self.prelu = nn.PReLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=source_num * encoded_frame_dim,
            kernel_size=kernel_size,
        )

        self.source_num = source_num
        self.encoded_frame_dim = encoded_frame_dim
        self.window_length = window_length

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=encoded_frame_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x, segmentation_padding):
        batch_size, D, K, S = x.size()
        # after prelu/conv x: [batch_size, CE, K, S]
        x = self.prelu(x)
        x = self.conv1(x)
        # after reshape/permute x: [batch_size, C, E, S, K]
        x = x.reshape(batch_size, self.source_num, -1, K, S)
        x = x.permute(0, 1, 2, 4, 3)
        # after overlap_and_add x: [batch_size, C, E, L]
        x = overlap_and_add(x, self.window_length)
        if segmentation_padding[1]:
            x = x[:, :, :, segmentation_padding[0]:-segmentation_padding[1]]
        else:
            x = x[:, :, :, segmentation_padding[0]:]

        # after reshape x: [batch_size * C, E, L]
        x = x.reshape(batch_size * self.source_num, self.encoded_frame_dim, -1)
        L = x.size()[2]
        x = self.relu(x)
        # after conv2 x: [batch_size * C, M, L]
        x = self.conv2(x)
        x = x.reshape(batch_size, self.source_num, -1, L)
        return x


class Decoder(nn.Module):
    def __init__(self, window_length):
        super(Decoder, self).__init__()
        self.window_length = window_length

    def forward(self, x, x_wave, encoder_padding):
        # input x: [batch_size, C, M, L]
        # after multiplication x: [batch_size, C, M, L]
        x_wave = x_wave.unsqueeze(dim=1)
        x = x * x_wave
        # after permute x: [batch_size, C, L, M]
        x = x.permute(0, 1, 3, 2)
        # after overlap_and_add x: [batch_size, C, T]
        x = overlap_and_add(x, self.window_length)
        if encoder_padding[1]:
            x = x[:, :, encoder_padding[0]:-encoder_padding[1]]
        else:
            x = x[:, :, encoder_padding[0]:]

        return x


class Sandglasset(nn.Module):
    def __init__(
        self,
        encoder_window_length=4,  # M = 4
        encoder_out_channels=256, # E = 256
        encoder_out_features=128, # D = 128
        encoder_kernel_size=1,
        segment_length=256,       # K = 256
        sandglasset_block_num=6,  # N = 6
        hidden_size=128,          # H = 128
        num_layers=1,
        num_heads=8,              # J = 8
        dropout=0.1,
        mask_estimation_kernel_size=1,
        source_num=1,
    ):
        super(Sandglasset, self).__init__()
        self.tasnet_encoder = TasNetEncoder(
            window_length=encoder_window_length,
            out_channels=encoder_out_channels,
            out_features=encoder_out_features,
            kernel_size=encoder_kernel_size,
        )

        self.segmentation = SegmentationModule(segment_length=segment_length)
        self.sandglasset_blocks_num = sandglasset_block_num

        sandglasset_blocks = []
        for i in range(sandglasset_block_num):
            sandglasset_blocks.append(
                SandglassetBlock(
                    encoder_out_features,
                    4**(i if i < sandglasset_block_num // 2 else sandglasset_block_num - i - 1),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        self.sandglasset_blocks = nn.ModuleList(sandglasset_blocks)
        self.mask_estimation = MaskEstimation(
            encoder_out_features,
            encoder_window_length,
            source_num,
            encoder_out_channels,
            segment_length // 2,
            kernel_size=mask_estimation_kernel_size,
        )

        self.decoder = Decoder(encoder_window_length // 2)


    def forward(self, x):
        x, x_wave, encoder_padding = self.tasnet_encoder(x)
        x, segmentation_padding = self.segmentation(x)

        x_residuals = []
        for i in range(self.sandglasset_blocks_num):
            if i < self.sandglasset_blocks_num // 2:
                x = self.sandglasset_blocks[i](x)
                x_residuals.append(x)
            else:
                x = self.sandglasset_blocks[i](x + x_residuals[self.sandglasset_blocks_num - i - 1])

        x = self.mask_estimation(x, segmentation_padding)
        x = self.decoder(x, x_wave, encoder_padding)

        return x
