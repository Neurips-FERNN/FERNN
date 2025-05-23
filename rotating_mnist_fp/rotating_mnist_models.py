import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from escnn import gspaces, nn as e2nn              # `escnn` ≥ 0.1



import math

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
            decoder += [nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU()]
        decoder += [nn.Conv2d(hidden_channels, self.output_channels, 3, padding=1)]
        self.decoder_conv = nn.Sequential(*decoder)

    def forward(self, input_seq, pred_len, teacher_forcing_ratio=1.0, target_seq=None):
        batch, T_in, C, H, W = input_seq.size()
        device = input_seq.device

        # Initialize hidden
        h = torch.zeros(batch * self.num_v, self.hidden_channels, H, W, device=device)

        # Encoder pass
        for t in range(T_in):
            u_t = input_seq[:, t]
            h = self.cell(u_t, h)

        prev = input_seq[:, -1]
        outputs = []

        # Decoder pass
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                frame = target_seq[:, t]
            else:
                frame = prev.detach()

            h = self.cell(frame, h)

            # Pool across velocities
            h_sep = h.view(batch, self.num_v, self.hidden_channels, H, W)
            if self.pool_type == 'max':
                feat = h_sep.max(1)[0]
            elif self.pool_type == 'mean':
                feat = h_sep.mean(1)
            elif self.pool_type == 'sum':
                feat = h_sep.sum(1)
            else:
                feat = h_sep.max(1)[0]

            out_frame = self.decoder_conv(feat)
            outputs.append(out_frame)
            prev = out_frame

        return torch.stack(outputs, dim=1)



def build_rot_equivariant_conv(in_type, out_channels, kernel_size):
    """Return an R2Conv that is equivariant to the discrete rotation group."""
    gspace = in_type.gspace                       # same g-space
    out_type = e2nn.FieldType(
        gspace, out_channels * [gspace.regular_repr]
    )
    padding = kernel_size // 2                    # keep spatial size
    conv = e2nn.R2Conv(in_type, out_type,
                       kernel_size=kernel_size,
                       padding=padding, bias=False)
    return conv, out_type


def rotate_and_permute(tensor, k, N):
    """
    tensor : (B, C, H, W) with C = hidden_channels * N
    k      : integer rotation step  (positive = counter-clockwise)
    N      : group order
    """
    # 1) cyclically permute orientation channels (regular repr.)
    B, C, H, W = tensor.shape
    C0 = C // N
    tensor = tensor.view(B, C0, N, H, W)          # split orientation axis
    tensor = torch.roll(tensor, shifts=k, dims=2) # shift δ_{g_i} → δ_{g_{i+k}}
    tensor = tensor.view(B, C, H, W)

    # 2) rotate the spatial grid by the same angle
    if k % 4 == 0:                                # multiples of 90° — cheap path
        return torch.rot90(tensor, k // 2, dims=(2, 3))

    angle = 2 * math.pi * k / N
    rot_mat = tensor.new_tensor([[ math.cos(angle), -math.sin(angle), 0],
                                 [ math.sin(angle),  math.cos(angle), 0]])
    grid = F.affine_grid(rot_mat.unsqueeze(0), tensor.size(),
                         align_corners=False)
    return F.grid_sample(tensor, grid, align_corners=False,
                         padding_mode='circular')

# -- single library call that applies BOTH channel perm + spatial rot ----
def act_on_hidden(self, x, w):
        """
        x : (B, C, H, W) tensor, C = hid_ch * N
        ω : integer angular-velocity step
        """
        import ipdb; ipdb.set_trace()
        g_idx = int(w * self.N / 360)
        g_el = self.gspace.fibergroup.elements[g_idx]  # group element
        gt   = e2nn.GeometricTensor(x, self.hid_type)
        return gt.transform(g_el).tensor                        # <- one call!


class RotationFlowRNNCell(nn.Module):
    """
    One step of a rotation-flow equivariant FERNN.
    The hidden state has shape (B, num_ω, hidden, H, W).
    """
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3,
                 v_list=[-40, -20, 0, 20, 40], N=8):
        super().__init__()
        # --- group & field types ------------------------------------------------
        self.N = N
        self.gspace      = gspaces.rot2dOnR2(N=N)        # C_N group
        in_type_u        = e2nn.FieldType(self.gspace,
                                          input_channels * [self.gspace.trivial_repr])
        self.in_type_u   = in_type_u
        self.v_list      = v_list  # list of angular velocities
        self.num_v       = len(self.v_list)

        # input to hidden ---------------------------------------------------------
        self.conv_u, hid_type = build_rot_equivariant_conv(in_type_u,
                                                           hidden_channels,
                                                           u_kernel_size)
        self.hid_type = hid_type

        # hidden to hidden --------------------------------------------------------
        self.conv_h, _ = build_rot_equivariant_conv(hid_type,
                                                    hidden_channels,
                                                    h_kernel_size)
        self.hidden_channels = hidden_channels
        self.activation      = nn.ReLU()

    # -------------------------------------------------------------------------
    def act_on_hidden(self, tensor, k):
        """
        tensor : (B, C, H, W) with C = hidden_channels * N
        k      : angle to rotate in degrees
        N      : group order
        """
        # 1) cyclically permute orientation channels (regular repr.)
        B, C, H, W = tensor.shape
        C0 = C // self.N
        tensor = tensor.view(B, C0, self.N, H, W)          # split orientation axis
        g_idx = int(k * self.N / 360) 
        tensor = torch.roll(tensor, shifts=g_idx, dims=2) # shift d_{g_i} → d_{g_{i+k}}
        tensor = tensor.view(B, C, H, W)

        # 2) rotate the spatial grid by the same angle
        if k % 90 == 0:                                # multiples of 90deg — cheap path
            return torch.rot90(tensor, k // 2, dims=(2, 3))

        # Convert degrees to radians
        angle_rad = math.radians(k)
        rot_mat = tensor.new_tensor([[ math.cos(angle_rad), -math.sin(angle_rad), 0],
                                    [ math.sin(angle_rad),  math.cos(angle_rad), 0]])
        # Expand rotation matrix to match batch size
        rot_mat = rot_mat.unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(rot_mat, tensor.size(),
                            align_corners=False)
        return F.grid_sample(tensor, grid, align_corners=False,
                            padding_mode='zeros')


    # -------------------------------------------------------------------------
    def forward(self, u_t, h_prev):
        """
        u_t     : (B, C, H, W)
        h_prev  : (B, num_ω, hidden, H, W)
        returns : h_next with same shape as h_prev
        """
        B, _, H, W = u_t.shape
        # lift input frame to g-space ------------------------------------------
        u_feat = e2nn.GeometricTensor(u_t, self.in_type_u)
        u_conv = self.conv_u(u_feat).tensor                # (B, hidden, H, W)
        u_conv = u_conv.unsqueeze(1).expand(-1, self.num_v, -1, -1, -1)

        # rotate hidden slices by their w --------------------------------------
        h_rot = []
        for i, w in enumerate(self.v_list):
            h_slice = h_prev[:, i]                         # (B, hidden, H, W)
            h_rot.append(self.act_on_hidden(h_slice, w))
        h_rot = torch.stack(h_rot, dim=1)                  # (B, num_ω, hidden, H, W)

        # group-equivariant convolution on each slice --------------------------
        h_flat = h_rot.reshape(B * self.num_v, self.hidden_channels * self.N, H, W)
        h_conv = self.conv_h(e2nn.GeometricTensor(h_flat,
                                                  self.conv_h.in_type)).tensor
        h_conv = h_conv.view(B, self.num_v, self.hidden_channels * self.N, H, W)

        # recurrence -----------------------------------------------------------
        h_next = self.activation(u_conv + h_conv)
        return h_next


class Seq2SeqRotationRNN(nn.Module):
    """
    Rotation-flow equivariant sequence-to-sequence model.
    Matches the I/O signature of your original Galilean model.
    """
    def __init__(self, input_channels, hidden_channels,
                 height, width, output_channels=None,
                 h_kernel_size=3, u_kernel_size=3,
                 v_list=[-40, -20, 0, 20, 40], N=8, pool_type='max',
                 decoder_conv_layers=1):
        super().__init__()
        self.height, self.width = height, width
        self.pool_type          = pool_type
        self.output_channels    = output_channels or input_channels

        self.cell = RotationFlowRNNCell(input_channels, hidden_channels,
                                         h_kernel_size, u_kernel_size,
                                         v_list=v_list, N=N)
        self.hidden_channels = hidden_channels
        self.total_hidden_channels = hidden_channels * N
        self.num_v = self.cell.num_v
        self.N = N

        # decoder (ordinary channelwise convs – rotation invariance achieved by pooling) –
        layers = []
        for _ in range(decoder_conv_layers):
            layers += [nn.Conv2d(self.total_hidden_channels,
                                 self.total_hidden_channels,
                                 kernel_size=3, padding=1, bias=False),
                        nn.ReLU()]
        layers += [nn.Conv2d(self.total_hidden_channels,
                             self.output_channels,
                             kernel_size=3, padding=1, bias=False)]
        self.decoder_conv = nn.Sequential(*layers)

    # -------------------------------------------------------------------------
    def forward(self, input_seq, pred_len, teacher_forcing_ratio=1.0,
                target_seq=None, return_hidden=False):
        """
        input_seq : (B, T_in, C, H, W)
        returns   : (B, pred_len, C_out, H, W)
        """
        B, T_in, C, H, W = input_seq.shape
        device = input_seq.device

        h = torch.zeros(B, self.num_v, self.total_hidden_channels,
                        H, W, device=device)

        # encoder -------------------------------------------------------------
        for t in range(T_in):
            h = self.cell(input_seq[:, t], h)

        prev = input_seq[:, -1]
        outputs = []
        hiddens = []

        # decoder -------------------------------------------------------------
        for t in range(pred_len):
            frame = (target_seq[:, t] if self.training and
                     target_seq is not None and
                     random.random() < teacher_forcing_ratio else prev.detach())
            h = self.cell(frame, h)

            if return_hidden:
                hiddens.append(h.clone())

            # pool over w (velocity) dimension – yields rotation-invariant code
            if   self.pool_type == 'mean': feat = h.mean(1)
            elif self.pool_type == 'sum':  feat = h.sum(1)
            else:                          feat = h.max(1)[0]

            out = self.decoder_conv(feat)
            outputs.append(out)
            prev = out

        if return_hidden:
            return torch.stack(outputs, dim=1), hiddens
        else:
            return torch.stack(outputs, dim=1)