import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class Seq2SeqStandardRNN(nn.Module):
    """
    Sequence-to-sequence RNN for forward prediction of Moving MNIST.
    Supports deep encoder, multi-layer recurrence, and deep decoder.

    Args:
        input_channels, height, width: frame dimensions
        hidden_size: size of RNN hidden state
        output_channels: channels in output frame (defaults to input_channels)
        num_layers: number of stacked RNNCell layers
        decoder_hidden_layers: number of hidden linear layers in decoder (with ReLU)
    """
    def __init__(
        self,
        input_channels,
        height,
        width,
        hidden_size,
        output_channels=None,
        num_layers=1,
        decoder_hidden_layers=1,
    ):
        super().__init__()
        self.input_size = input_channels * height * width
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.height = height
        self.width = width
        self.output_channels = output_channels or input_channels

        # encoder / decoder RNN cells
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(
                self.input_size if i == 0 else hidden_size,
                hidden_size
            ) for i in range(num_layers)
        ])

        # deep decoder: linear layers then output
        decoder_layers = []
        for i in range(decoder_hidden_layers):
            decoder_layers.append(nn.Linear(
                hidden_size, hidden_size
            ))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(
            hidden_size,
            self.output_channels * height * width
        ))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(
        self,
        input_seq,
        pred_len,
        teacher_forcing_ratio=1.0,
        target_seq=None,
    ):
        """
        Args:
            input_seq: (batch, T_in, C, H, W)
            pred_len: number of frames to predict
            teacher_forcing_ratio: probability of using ground-truth as input during decoding
            target_seq: (batch, pred_len, C, H, W), required if teacher_forcing_ratio > 0
        Returns:
            outputs: (batch, pred_len, C, H, W)
        """
        batch, T_in, C, H, W = input_seq.size()
        device = input_seq.device

        # flatten inputs
        input_flat = input_seq.view(batch, T_in, -1)

        # init hidden states
        h = [torch.zeros(batch, self.hidden_size, device=device)
             for _ in range(self.num_layers)]

        # encoder pass over input sequence
        for t in range(T_in):
            x = input_flat[:, t]
            for l, cell in enumerate(self.rnn_cells):
                h[l] = cell(x, h[l])
                x = h[l]

        # decoder
        prev_frame = input_seq[:, -1]
        outputs = []
        for t in range(pred_len):
            if target_seq is not None and random.random() < teacher_forcing_ratio and self.training:
                x = target_seq[:, t]
            else:
                x = prev_frame
            x = x.view(batch, -1)
            # propagate through rnn cells
            for l, cell in enumerate(self.rnn_cells):
                h[l] = cell(x, h[l])
                x = h[l]
            # decode to frame
            out_flat = self.decoder(x)
            out_frame = out_flat.view(batch, self.output_channels, H, W)
            outputs.append(out_frame)
            prev_frame = out_frame

        return torch.stack(outputs, dim=1)


class ConvRNNCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvRNNCell, self).__init__()
        padding = kernel_size // 2  # To maintain spatial dimensions
        self.conv_x = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, padding_mode='circular', bias=False)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode='circular', bias=False)
        self.activation = nn.ReLU()
        
    def forward(self, x, h_prev):
        h = self.activation(self.conv_x(x) + self.conv_h(h_prev))
        return h



class Seq2SeqConvRNN(nn.Module):
    """
    Sequence-to-sequence Conv-RNN for Moving MNIST forward prediction.
    Supports deep encoder (stacked ConvRNNCell), Conv decoder layers, and autoregressive/teacher-forced decoding.

    Args:
        input_channels, hidden_channels: feature channels
        height, width: frame size
        output_channels: output frame channels (defaults to input_channels)
        num_layers: number of stacked ConvRNNCell layers
        encoder_kernel_size: kernel size for ConvRNNCell
        decoder_conv_layers: number of extra Conv2d+ReLU layers in decoder
    """
    def __init__(
        self,
        input_channels,
        hidden_channels,
        height,
        width,
        output_channels=None,
        num_layers=1,
        encoder_kernel_size=3,
        decoder_conv_layers=1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels or input_channels

        # encoder ConvRNNCells
        self.rnn_cells = nn.ModuleList([
            ConvRNNCell(
                input_channels if i == 0 else hidden_channels,
                hidden_channels,
                encoder_kernel_size
            ) for i in range(num_layers)]
        )

        # deep Conv decoder
        decoder_layers = []
        for i in range(decoder_conv_layers):
            decoder_layers.append(nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ))
            decoder_layers.append(nn.ReLU())
        # final conv to output channels
        decoder_layers.append(nn.Conv2d(
            hidden_channels,
            self.output_channels,
            kernel_size=3,
            padding=1
        ))
        self.decoder_conv = nn.Sequential(*decoder_layers)

    def forward(
        self,
        input_seq,
        pred_len,
        teacher_forcing_ratio=1.0,
        target_seq=None,
    ):
        """
        Args:
            input_seq: (batch, T_in, C, H, W)
            pred_len: number of frames to predict
            teacher_forcing_ratio: probability of using ground-truth at each step
            target_seq: (batch, pred_len, C, H, W)
        Returns:
            outputs: (batch, pred_len, C, H, W)
        """
        batch, T_in, C, H, W = input_seq.size()
        device = input_seq.device

        # init hidden states
        h = [torch.zeros(batch, self.hidden_channels, H, W, device=device)
             for _ in range(self.num_layers)]

        # encoder pass
        for t in range(T_in):
            x = input_seq[:, t]
            for l, cell in enumerate(self.rnn_cells):
                h[l] = cell(x, h[l])
                x = h[l]

        # decoder pass
        prev = input_seq[:, -1]
        outputs = []
        for t in range(pred_len):
            if target_seq is not None and random.random() < teacher_forcing_ratio and self.training:
                x = target_seq[:, t]
            else:
                x = prev
            for l, cell in enumerate(self.rnn_cells):
                h[l] = cell(x, h[l])
                x = h[l]
            out = self.decoder_conv(x)
            outputs.append(out)
            prev = out

        return torch.stack(outputs, dim=1)



class ConvRNNCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvRNNCell, self).__init__()
        padding = kernel_size // 2  # To maintain spatial dimensions
        self.conv_x = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, padding_mode='circular', bias=False)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode='circular', bias=False)
        self.activation = nn.ReLU()
        
    def forward(self, x, h_prev):
        h = self.activation(self.conv_x(x) + self.conv_h(h_prev))
        return h



class Seq2SeqConvRNN(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 height,
                 width,
                 output_channels=None,
                 num_layers=1,
                 encoder_kernel_size=3,
                 decoder_conv_layers=1,
                 group_equiv=False,
                 num_v=1,
                 pool_type='max'):
        super().__init__()
        self.num_layers = num_layers
        self.group_equiv = group_equiv
        self.num_v = num_v
        self.pool_type = pool_type
        self.base_hidden = hidden_channels
        self.hidden_channels = hidden_channels * num_v if group_equiv else hidden_channels
        self.height = height
        self.width = width
        self.output_channels = output_channels or input_channels

        # Encoder ConvRNNCells
        self.rnn_cells = nn.ModuleList([
            ConvRNNCell(
                input_channels if i == 0 else self.hidden_channels,
                self.hidden_channels,
                encoder_kernel_size
            ) for i in range(num_layers)]
        )

        # Decoder conv: input channels depends on grouping
        in_ch = self.base_hidden if group_equiv else self.hidden_channels
        decoder = []
        for _ in range(decoder_conv_layers):
            decoder += [nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False), nn.ReLU()]
        decoder += [nn.Conv2d(in_ch, self.output_channels, 3, padding=1, bias=False)]
        self.decoder_conv = nn.Sequential(*decoder)

    def forward(self, input_seq, pred_len, teacher_forcing_ratio=1.0, target_seq=None):
        batch, T_in, C, H, W = input_seq.size()
        device = input_seq.device
        h = [torch.zeros(batch, self.hidden_channels, H, W, device=device)
             for _ in range(self.num_layers)]

        # Encoder
        for t in range(T_in):
            x = input_seq[:, t]
            for i, cell in enumerate(self.rnn_cells):
                h[i] = cell(x, h[i])
                x = h[i]

        prev = input_seq[:, -1]
        outputs = []

        # Decoder
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                frame = target_seq[:, t]
            else:
                frame = prev.detach()
            x = frame
            for i, cell in enumerate(self.rnn_cells):
                h[i] = cell(x, h[i])
                x = h[i]
            out = x  # (batch, hidden_channels, H, W)

            # Pool if group equivariant
            if self.group_equiv:
                out_sep = out.view(batch, self.num_v, self.base_hidden, H, W)
                if self.pool_type == 'max':
                    feat = out_sep.max(1)[0]
                elif self.pool_type == 'mean':
                    feat = out_sep.mean(1)
                elif self.pool_type == 'sum':
                    feat = out_sep.sum(1)
                else:
                    feat = out_sep.max(1)[0]
            else:
                feat = out
            out_frame = self.decoder_conv(feat)
            outputs.append(out_frame)
            prev = out_frame

        return torch.stack(outputs, dim=1)





class GalileanCovariantRNNCell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3, v_range=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)

        # circular convs without bias
        u_pad = u_kernel_size // 2
        h_pad = h_kernel_size // 2
        self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                 padding=u_pad, padding_mode='circular', bias=False)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                 padding=h_pad, padding_mode='circular', bias=False)
        self.activation = nn.ReLU()

    def forward(self, u_t, h_prev):
        # u_t: (batch, C, H, W)
        # h_prev: (batch, num_v, hidden, H, W)
        batch, C, H, W = u_t.size()
        # conv_u then expand
        u_conv = self.conv_u(u_t)  # (batch, hidden, H, W)
        u_conv = u_conv.unsqueeze(1).expand(-1, self.num_v, -1, -1, -1)

        # shift hidden via torch.roll per velocity
        h_shift = []
        for i, (vx, vy) in enumerate(self.v_list):
            h_shift.append(torch.roll(h_prev[:, i], shifts=(vy, vx), dims=(2, 3)))
        h_shift = torch.stack(h_shift, dim=1)  # (batch, num_v, hidden, H, W)

        # conv_h on flattened v dimension
        h_flat = h_shift.view(batch * self.num_v, self.hidden_channels, H, W)
        h_conv = self.conv_h(h_flat)
        h_conv = h_conv.view(batch, self.num_v, self.hidden_channels, H, W)

        # combine and activate
        h_next = self.activation(u_conv + h_conv)
        return h_next


class Seq2SeqGalileanRNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, height, width,
                 output_channels=None, h_kernel_size=3, u_kernel_size=3,
                 v_range=0, pool_type='max', decoder_conv_layers=1):
        super().__init__()
        self.height = height
        self.width = width
        self.pool_type = pool_type
        self.output_channels = output_channels or input_channels

        self.cell = GalileanCovariantRNNCell(
            input_channels, hidden_channels,
            h_kernel_size, u_kernel_size, v_range)
        self.hidden_channels = hidden_channels
        self.num_v = self.cell.num_v

        decoder = []
        for _ in range(decoder_conv_layers):
            decoder += [nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False), nn.ReLU()]
        decoder += [nn.Conv2d(hidden_channels, self.output_channels, 3, padding=1, bias=False)]
        self.decoder_conv = nn.Sequential(*decoder)

    def forward(self, input_seq, pred_len, teacher_forcing_ratio=1.0, target_seq=None, return_hidden=False):
        batch, T_in, C, H, W = input_seq.size()
        device = input_seq.device

        if return_hidden:
            input_seq_hiddens = torch.zeros(batch, T_in, self.num_v, self.hidden_channels, H, W, device=device)
            out_seq_hiddens = torch.zeros(batch, pred_len, self.num_v, self.hidden_channels, H, W, device=device)

        # Initialize hidden state
        h = torch.zeros(batch, self.num_v, self.hidden_channels, H, W, device=device)

        # Encoder pass through cell
        for t in range(T_in):
            u_t = input_seq[:, t]
            h = self.cell(u_t, h)

            if return_hidden:
                input_seq_hiddens[:, t] += h.detach()

        prev = input_seq[:, -1]
        outputs = []

        # Decoder
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                frame = target_seq[:, t]
            else:
                frame = prev.detach()
            h = self.cell(frame, h)

            if return_hidden:
                out_seq_hiddens[:, t] += h.detach()

            # pool over velocities
            if self.pool_type == 'max':
                feat = h.max(1)[0]
            elif self.pool_type == 'mean':
                feat = h.mean(1)
            elif self.pool_type == 'sum':
                feat = h.sum(1)
            else:
                feat = h.max(1)[0]

            out = self.decoder_conv(feat)
            outputs.append(out)
            prev = out

        if return_hidden:
            return torch.stack(outputs, dim=1), input_seq_hiddens, out_seq_hiddens # _, (B, T_in, num_v, C, H, W), (B, T_out, num_v, C, H, W)
        else:
            return torch.stack(outputs, dim=1)
