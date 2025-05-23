import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from moving_mnist_dataset import MovingMNISTDataset, FixedVelocityMovingMNIST
from moving_mnist_models import Seq2SeqStandardRNN, Seq2SeqConvRNN, Seq2SeqGalileanRNN
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision.utils import make_grid
import time
import os
import math
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, criterion, device, input_frames, teacher_forcing_ratio, grad_clip=None):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for i, (seq, _) in enumerate(pbar):
        seq = seq.to(device)  # (B, seq_len, C, H, W)
        input_seq = seq[:, :input_frames]
        target_seq = seq[:, input_frames:]
        pred_len = target_seq.size(1)

        optimizer.zero_grad() 
        output_seq = model(
            input_seq,
            pred_len=pred_len,
            teacher_forcing_ratio=teacher_forcing_ratio,
            target_seq=target_seq
        )  # (B, pred_len, C, H, W)
        loss = criterion(output_seq, target_seq)
        loss.backward()
        
        # Apply gradient clipping if specified
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss * seq.size(0)
        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        # Log images for visualization (first batch and every 5 epochs)
        if i == 0:
            log_sequence_predictions(input_seq, target_seq, output_seq, split_name="train")

    return running_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, criterion, device, input_frames, epoch, split_name):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for i, (seq, _) in enumerate(pbar):
            seq = seq.to(device)
            input_seq = seq[:, :input_frames]
            target_seq = seq[:, input_frames:]
            pred_len = target_seq.size(1)

            output_seq = model(
                input_seq,
                pred_len=pred_len,
                teacher_forcing_ratio=0.0
            )
            loss = criterion(output_seq, target_seq)
            batch_loss = loss.item()
            running_loss += batch_loss * seq.size(0)
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})
            
            # Log images for visualization (first batch and every 5 epochs)
            if i == 0:
                log_sequence_predictions(input_seq, target_seq, output_seq, split_name=split_name)

    return running_loss / len(dataloader.dataset)


def eval_len_generalization(model, dataloader, device, input_frames, subsample_t=1):
    """
    Returns:
        mean_err  – numpy array [T]  (MSE at each future step, averaged over test set)
        std_err   – numpy array [T]  (sample‑wise std at each step)
    """
    model.eval()
    first_pass = True
    with torch.no_grad():
        n_sequences = 0
        pbar = tqdm(dataloader, desc="Evaluating Length Generalization", leave=False)
        for seq, _ in pbar:
            seq = seq.to(device)
            inp, tgt = seq[:, :input_frames], seq[:, input_frames:]
            T = tgt.size(1)
            pred = model(inp, pred_len=T, teacher_forcing_ratio=0.0)

            # MSE per example per timestep  →  [B, T]
            per_ex_t = ((pred - tgt)**2).mean(dim=(2, 3, 4))  # assume (B,T,C,H,W)
            if first_pass:
                sum_err  = per_ex_t.sum(dim=0)          # [T]
                sum_err2 = (per_ex_t**2).sum(dim=0)     # [T]
                first_pass, T_global = False, T

                log_sequence_predictions_new(inp, tgt, pred, split_name="len_gen", num_samples=10, device=device, subsample_t=subsample_t)

            else:
                sum_err  += per_ex_t.sum(dim=0)
                sum_err2 += (per_ex_t**2).sum(dim=0)

            n_sequences += per_ex_t.size(0)
            
            # Update progress bar with current batch size
            pbar.set_postfix({"loss": per_ex_t.mean()})

    mean = sum_err / n_sequences
    var  = sum_err2 / n_sequences - mean**2
    std  = torch.sqrt(torch.clamp(var, min=0.0))
    return mean.cpu().numpy(), std.cpu().numpy()


def eval_velocity_generalization(model, device, args):
    """
    Returns
    -------
    vx_vals : np.ndarray [K]   (sorted unique velocities on x-axis)
    vy_vals : np.ndarray [K]   (same on y-axis)
    err_mat : np.ndarray [K,K] (mean MSE at (vy, vx))
    """
    vx_vals = np.arange(args.gen_vel_min, args.gen_vel_max + 1, args.gen_vel_step)
    vy_vals = np.arange(args.gen_vel_min, args.gen_vel_max + 1, args.gen_vel_step)
    err_mat = np.zeros((len(vy_vals), len(vx_vals)))

    crit = torch.nn.MSELoss(reduction='none')
    model.eval()

    vel_pbar = tqdm(total=len(vy_vals)*len(vx_vals), desc="Evaluating Velocity Generalization", leave=False)
    for iy, vy in enumerate(vy_vals):
        for ix, vx in enumerate(vx_vals):
            vel_pbar.set_postfix({"vx": vx, "vy": vy})
            dataset = FixedVelocityMovingMNIST(
                vx=vx, vy=vy,
                root=args.root,
                train=False,
                seq_len=args.seq_len,
                image_size=args.image_size,
                num_digits=1,
                random=False)
            # subset, _ = torch.utils.data.random_split(
            #     dataset, [args.gen_vel_n_seq, len(dataset) - args.gen_vel_n_seq],
            #     generator=torch.Generator().manual_seed(42))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            # loader = DataLoader(subset, batch_size=args.batch_size,
                                # shuffle=False)

            mse_sum, n_seen = 0.0, 0
            with torch.no_grad():
                batch_pbar = tqdm(loader, desc=f"vx={vx}, vy={vy}", leave=False)
                for seq, _ in batch_pbar:
                    seq = seq.to(device)
                    inp, tgt = seq[:, :args.input_frames], seq[:, args.input_frames:]
                    pred = model(inp, pred_len=tgt.size(1), teacher_forcing_ratio=0.0)
                    mse = crit(pred, tgt).mean(dim=(2, 3, 4))  # [B, T]
                    batch_mse = mse.mean().item()
                    mse_sum += batch_mse * mse.size(0)
                    n_seen  += mse.size(0)
                    batch_pbar.set_postfix({"mse": f"{batch_mse:.4f}"})
                    # Log sequence predictions for the first batch of this velocity pair
                    if batch_pbar.n == 0: # Check if it's the first batch
                        log_sequence_predictions_new(inp, tgt, pred, split_name=f"vel_gen_vx{vx}_vy{vy}", num_samples=10, device=device)
            err_mat[iy, ix] = mse_sum / n_seen
            vel_pbar.update(1)
    vel_pbar.close()

    return vx_vals, vy_vals, err_mat

def velocity_permutation(v_range: int, vx: int, vy: int) -> torch.Tensor:
    """
    Returns a 1-D LongTensor `perm` of length (2*v_range+1)**2 such that

        h_permuted = h.index_select(dim=1, index=perm)

    applies the ν ↦ (vx,vy)^(-1) ν permutation required by flow equivariance
    (see Eq. 17 in the paper).  Works for *any* tensor whose 2nd dim is num_v.
    """
    # build the reference order once
    v_list = [(x, y)
              for x in range(-v_range, v_range + 1)
              for y in range(-v_range, v_range + 1)]

    inv_v = (vx, vy)                         # group inverse in Z²
    lookup = {v: i for i, v in enumerate(v_list)}

    perm = [lookup[((vx0 + inv_v[0]) % v_range, (vy0 + inv_v[1]) % v_range)]
            for (vx0, vy0) in v_list]
    return torch.tensor(perm, dtype=torch.long).to('cuda')

# -------------------------------------------------------------------- #
#  Simple helper: spatially roll each frame by t·(vx,vy) (wrap-around) #
# -------------------------------------------------------------------- #
def _translate_sequence(x, vx, vy, start_time=0):
    """
    x : [B, T, C, H, W]
    h: [B, T, num_v, C, H, W]
        returns a version where frame t is rolled by (t*vy, t*vx)
    """
    # b, t, c, h, w = x.shape
            
    t = x.shape[1]
    out = torch.empty_like(x)
    for step in range(t):

        y_roll = int((step + start_time) * vy)
        x_roll = int((step + start_time) * vx)

        out[:, step] = torch.roll(
            x[:, step],
            shifts=(y_roll, x_roll),   # (y, x)
            dims=(-2, -1)                              # H , W
        )

    if len(x.shape) == 6:
        # perm = velocity_permutation(2, vx, vy)

        # out = out.index_select(dim=2, index=perm)
        # out = permute_velocity_channels(out, 2, vx, vy)
        B, T, num_v, C, H, W = x.shape
        if num_v > 1:
            square_size = int(np.sqrt(num_v))
            out = out.view(B, T, square_size, square_size, C, H, W)
            out = torch.roll(out, shifts=(int(vx), int(vy)), dims=(2, 3))
            out = out.view(B, T, num_v, C, H, W)

    return out


def log_sequence_predictions(
        input_seq, target_seq, output_seq,
        split_name,
        num_samples=2,
        frames_per_row=10,         
        upsample_scale=4,          
        dpi=160                   
    ):
    """
    Visualise GT / prediction / |diff| for a handful of samples.

    • frames_per_row   controls the wrapping, keeping height reasonable even
                       for very long sequences.
    • upsample_scale   multiplies the resolution of every frame to make small
                       MNIST digits clearly visible.
    """
    batch_size = input_seq.size(0)
    num_samples = min(num_samples, batch_size)
    indices = np.random.choice(batch_size, num_samples, replace=False)

    input_len  = input_seq.size(1)
    target_len = target_seq.size(1)
    total_len  = input_len + target_len

    # grid layout parameters ----------------------------------------------------
    ncols = min(frames_per_row, total_len)           # frames per grid‐row
    nrows = math.ceil(total_len / ncols)             # how many rows per grid
    # --------------------------------------------------------------------------

    # figure size in *inches*: width ~ ncols * upsample_scale * 0.25
    fig_w = (ncols * upsample_scale) * 0.25
    fig_h = (3 * nrows * upsample_scale) * 0.25      # 3 rows (GT / pred / diff)

    fig, axes = plt.subplots(
        3, num_samples,
        figsize=(fig_w * num_samples, fig_h),
        dpi=dpi,
        squeeze=False
    )

    for i, idx in enumerate(indices):
        s_in   = input_seq[idx].cpu()
        s_tgt  = target_seq[idx].cpu()
        s_pred = output_seq[idx].cpu()

        # full sequences --------------------------------------------------------
        full_gt   = torch.cat([s_in,  s_tgt],  dim=0)
        full_pred = torch.cat([s_in,  s_pred], dim=0)
        full_diff = torch.cat([torch.zeros_like(s_in),
                               torch.abs(s_pred - s_tgt)], dim=0)

        for row, tensor, title in zip(
                range(3),
                (full_gt, full_pred, full_diff),
                ("Ground Truth", "Prediction", "Difference |Δ|")):

            grid = make_grid(
                tensor, nrow=ncols, normalize=True, padding=1
            )

            # upscale the whole grid so each digit is bigger -------------------
            grid = F.interpolate(
                grid.unsqueeze(0),  # [1, C, H, W]
                scale_factor=upsample_scale,
                mode='nearest'
            ).squeeze(0)

            axes[row, i].imshow(grid.permute(1, 2, 0).numpy(),
                                interpolation='nearest')
            axes[row, i].set_title(f"Sample {i+1} – {title}",
                                   fontsize=10)
            axes[row, i].axis('off')

    plt.tight_layout()
    wandb.log({f"{split_name}_sequences": wandb.Image(fig)})
    plt.close(fig)

def log_sequence_predictions_new(
    input_seq, target_seq, output_seq,
    split_name,    
    num_samples: int = 4,          # number of sequences to visualise
    vmax_diff: float = 1.0,        # clip range for the signed difference plot
    subsample_t: int = 1,          # subsample the time dimension by this factor
    device: torch.device | None = None,
):
    """
    Visualise ground–truth, prediction, and signed error for a handful of sequences.

    """
    T = target_seq.shape[1]
    T = T // subsample_t

    # --- iterate over the first num_samples sequences ------------------------
    for idx in range(num_samples):
        gt_seq   = target_seq[idx].detach().cpu().squeeze()       # (T, H, W)
        pred_seq = output_seq[idx].detach().cpu().squeeze()   # (T, H, W)
        diff_seq = pred_seq - gt_seq                     # signed error

        # ----------- set up a long thin figure --------------------------------
        fig_height = 3          # one row per line, in inches
        fig_width  = max(6, T)
        fig, axes  = plt.subplots(
            3, T,
            figsize=(fig_width, fig_height),
            gridspec_kw={"wspace": 0.005, "hspace": 0.03},  # Reduced spacing between elements
        )

        # make axes always iterable in both dims
        if T == 1:
            axes = axes.reshape(3, 1)

        # ----------- plot -----------------------------------------------------
        for t in range(T):
            # top row – ground truth
            axes[0, t].imshow(gt_seq[t*subsample_t], cmap="gray", vmin=0, vmax=1)
            # centre row – predictions
            axes[1, t].imshow(pred_seq[t*subsample_t], cmap="gray", vmin=0, vmax=1)
            # bottom row – signed difference
            axes[2, t].imshow(
                diff_seq[t*subsample_t],
                cmap="bwr",
                vmin=-vmax_diff,
                vmax=vmax_diff,
            )

            # cosmetic clean-up
            for r in range(3):
                axes[r, t].axis("off")

        # label the rows once (left-most subplot)
        axes[0, 0].set_ylabel("GT",    rotation=0, labelpad=20, fontsize=10)
        axes[1, 0].set_ylabel("Pred",  rotation=0, labelpad=15, fontsize=10)
        axes[2, 0].set_ylabel("Error", rotation=0, labelpad=18, fontsize=10)

        # optional overall title
        fig.suptitle(f"{split_name} sample {idx}", fontsize=12)

        # ----------- log to wandb & close -------------------------------------
        wandb.log({f"{split_name}sequence_{idx}": wandb.Image(fig)})
        plt.close(fig)

def seed_worker(worker_id):
    # derive a unique seed for each worker from the base seed
    worker_seed = args.data_seed + worker_id
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def main():
    parser = argparse.ArgumentParser(description="Train & evaluate RNN models on Moving MNIST")
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--min_epochs', type=int, default=50, help='Minimum number of epochs to train')
    parser.add_argument('--total_train_steps', type=int, default=None, help='Total number of training steps (overrides epochs if not None)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', choices=['standard','conv', 'gal','both'], default='both')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--decoder_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--decoder_conv_layers', type=int, default=4)
    parser.add_argument('--group_equiv_convrnn', action='store_true', default=False)
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use (default: use all)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help='Probability of using teacher forcing during training (0.0-1.0)')
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--v_range', type=int, default=2)
    parser.add_argument('--data_v_range', type=int, default=2)
    parser.add_argument('--gen_seq_len', type=int, default=40, help='Sequence length used **only** for length‑generalization evaluation (must be > seq_len)')
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for dataset splitting')
    parser.add_argument('--model_seed', type=int, default=None, help='Random seed for model initialization (default: random)')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the run')
    parser.add_argument('--model_save_dir', type=str, default='./fernn/movmnist/', help='Directory to save model checkpoints')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model checkpoint to load for evaluation')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate the loaded model without training')
    parser.add_argument('--gen_vel_min', type=int, default=-2, help='Minimum integer pixel velocity evaluated along each axis')
    parser.add_argument('--gen_vel_max', type=int, default= 2, help='Maximum integer pixel velocity evaluated along each axis')
    parser.add_argument('--gen_vel_step', type=int, default= 1, help='Step size (integer) for the velocity grid')
    parser.add_argument('--gen_vel_n_seq', type=int, default=128, help='How many sequences to sample **per (vx,vy)** for generalisation test')
    parser.add_argument('--run_velocity_generalization', action='store_true', help='Run velocity generalization test (default: False)')
    parser.add_argument('--run_equivariance_loss', action='store_true', help='Run equivariance loss test (default: False)')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping value (default: None, no clipping)')
    args = parser.parse_args()

    # Set seeds for reproducibility
    # Data seed for consistent dataset splits
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)
    
    # Create model save directory if it doesn't exist
    os.makedirs(args.model_save_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(entity="ENTITY", 
               project="PROJECT", 
               dir='./tmp/',
               config=vars(args),
               name=f"moving_mnist_fp_{args.model}_{args.run_name}")

    assert args.input_frames < args.seq_len, "input_frames must be less than seq_len"
    pred_frames = args.seq_len - args.input_frames
    gen_pred_frames = args.gen_seq_len - args.input_frames

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and splits
    train_dataset = MovingMNISTDataset(
        root=args.root,
        train=True,
        seq_len=args.seq_len,
        image_size=args.image_size,
        velocity_range_x=(-args.data_v_range,args.data_v_range),
        velocity_range_y=(-args.data_v_range,args.data_v_range),
        num_digits=1
    )
    
    test_dataset = MovingMNISTDataset(
        root=args.root,
        train=False,
        seq_len=args.seq_len,
        image_size=args.image_size,
        velocity_range_x=(-args.data_v_range,args.data_v_range),
        velocity_range_y=(-args.data_v_range,args.data_v_range),
        num_digits=1
    )

    gen_test_dataset = MovingMNISTDataset(
            root=args.root,
            train=False,
            seq_len=args.gen_seq_len,
            image_size=args.image_size,
            velocity_range_x=(-args.data_v_range, args.data_v_range),
            velocity_range_y=(-args.data_v_range, args.data_v_range),
            num_digits=1,
            random=False
    )

    # Split training data into train and validation
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(args.data_seed))
    
    # Limit training samples if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_ds):
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.data_seed))[:args.max_train_samples]
        train_ds = Subset(train_ds, indices)
        print(f"Limited training to {args.max_train_samples} samples")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    gen_test_loader = DataLoader(gen_test_dataset,
                                batch_size=args.batch_size)


    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_dataset)}")
    print(f"Length‑gen test size: {len(gen_test_dataset)} | Eval sequence length: {args.gen_seq_len}")
    print(f"Input frames: {args.input_frames}, Pred frames: {pred_frames}")
    print(f"Model seed: {args.model_seed}")
    print(f"Data seed: {args.data_seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Min epochs: {args.min_epochs}")
    print(f"Total train steps: {args.total_train_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Model: {args.model}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Num layers: {args.num_layers}")
    print(f"Decoder layers: {args.decoder_layers}")
    print(f"Decoder conv layers: {args.decoder_conv_layers}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Teacher forcing ratio: {args.teacher_forcing_ratio}")
    print(f"Max train samples: {args.max_train_samples}")
    print(f"Image size: {args.image_size}")
    print(f"Velocity range: {args.v_range}")
    print(f"Group equiv convrnn: {args.group_equiv_convrnn}")
    v_list = [(x, y) for x in range(-args.v_range, args.v_range + 1) for y in range(-args.v_range, args.v_range + 1)]
    num_v = len(v_list)
    print(f"Num v: {num_v}")

    # Set model initialization seed
    if args.model_seed is None:
        args.model_seed = int(time.time()) % 10000  # Use current time as seed if not provided
        wandb.config.update({"model_seed": args.model_seed}, allow_val_change=True)
    
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    random.seed(args.model_seed)
    
    # Models
    models = {}
    if args.model in ('standard','both'):
        models['standard'] = Seq2SeqStandardRNN(
            input_channels=1,
            height=args.image_size,
            width=args.image_size,
            hidden_size=args.hidden_size,
            output_channels=1,
            num_layers=args.num_layers,
            decoder_hidden_layers=args.decoder_layers
        ).to(device)
    if args.model in ('conv','both'):
        models['conv'] = Seq2SeqConvRNN(
            input_channels=1,
            hidden_channels=args.hidden_size,
            height=args.image_size,
            width=args.image_size,
            output_channels=1,
            num_layers=args.num_layers,
            encoder_kernel_size=args.kernel_size,
            decoder_conv_layers=args.decoder_conv_layers,
            group_equiv=args.group_equiv_convrnn,
            num_v=num_v,
            pool_type='max'
        ).to(device)
    if args.model in ('gal','both'):
        assert args.num_layers == 1, "Galilean RNN only supports 1 layer"
        models['gal'] = Seq2SeqGalileanRNN(
            input_channels=1,
            hidden_channels=args.hidden_size,
            height=args.image_size,
            width=args.image_size,
            h_kernel_size=args.kernel_size,
            u_kernel_size=args.kernel_size,
            v_range=args.v_range,
            pool_type='max',
            decoder_conv_layers=args.decoder_conv_layers
        ).to(device)

    # Load model if specified
    if args.load_model is not None:
        print(f"Loading model from {args.load_model}")
        for name, model in models.items():
            try:
                model.load_state_dict(torch.load(args.load_model))
                print(f"Successfully loaded model weights for {name}")
            except Exception as e:
                print(f"Failed to load model weights for {name}: {e}")
                continue

    # If evaluate_only is True, skip training and only run evaluation
    if args.evaluate_only:
        print("Running evaluation only...")
        criterion = nn.MSELoss()
        for name, model in models.items():
            print(f"\nEvaluating {name} model:")
            # test_loss = eval_epoch(model, test_loader, criterion, device, args.input_frames, 0, split_name="test")
            # print(f"Test Loss: {test_loss:.4f}")

            # wandb.log({
                # f"test_loss": test_loss,
                # "epoch": 0
            # })

            print("Running length generalization test...")
            gen_mean, gen_std = eval_len_generalization(model, gen_test_loader, device, args.input_frames, subsample_t=2)
            print(f"Length generalization mean MSE: {gen_mean.mean():.4f}")

            wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
            wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
            wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})

            if args.run_velocity_generalization:
                print("Running velocity generalization test...")
                vx, vy, err = eval_velocity_generalization(model, device, args)
                print("Velocity generalization results:")
                for i, vx_i in enumerate(vx):
                    for j, vy_j in enumerate(vy):
                        print(f"Velocity (vx={vx_i}, vy={vy_j}): MSE {err[j, i]:.4f}")
                
                wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                        for i, vx_i in enumerate(vx)
                        for j, vy_j in enumerate(vy)})
                

                wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                        for i, vx_i in enumerate(vx)
                        for j, vy_j in enumerate(vy)})

                fig, ax = plt.subplots()
                im = ax.imshow(err[::-1],  # flip y so +vy is up
                            extent=[vx[0]-0.5, vx[-1]+0.5, vy[0]-0.5, vy[-1]+0.5],
                            origin='lower')
                ax.set_xlabel('$v_x$  (pixels / frame)')
                ax.set_ylabel('$v_y$')
                ax.set_title('Velocity-generalisation MSE')
                fig.colorbar(im, ax=ax)
                wandb.log({"vel_gen_heatmap": wandb.Image(fig)})
                plt.close(fig)

        return

    # Optimizers & loss
    optimizers = {name: torch.optim.Adam(m.parameters(), lr=args.lr)
                  for name, m in models.items()}
    criterion = nn.MSELoss()

    # Training loop
    history = {name: {'train_loss': [], 'val_loss': [], 'test_loss': []} for name in models}
    best_val_losses = {name: float('inf') for name in models}
    
    # Determine whether to use epochs or total_train_steps
    if args.total_train_steps is not None:
        # Calculate how many epochs we need to reach total_train_steps
        steps_per_epoch = len(train_loader)
        epochs_needed = (args.total_train_steps + steps_per_epoch - 1) // steps_per_epoch  # Ceiling division
        print(f"Training for {args.total_train_steps} steps (~{epochs_needed} epochs)")
        max_epochs = max(epochs_needed, args.min_epochs)  # Ensure we train for at least min_epochs
    else:
        max_epochs = args.epochs
    
    for epoch in range(1, max_epochs + 1):
        for name, model in models.items():
            train_loss = train_epoch(model, train_loader, optimizers[name], criterion, device, args.input_frames, args.teacher_forcing_ratio, args.grad_clip)
            val_loss = eval_epoch(model, val_loader, criterion, device, args.input_frames, epoch, split_name="val")
            history[name]['train_loss'].append(train_loss)
            history[name]['val_loss'].append(val_loss)
            print(f"Epoch {epoch}/{max_epochs} | {name:^8} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save model if it's the best so far
            if val_loss < best_val_losses[name]:
                best_val_losses[name] = val_loss
                model_filename = f"{name}_best_model_{wandb.run.id}.pth"
                model_path = os.path.join(args.model_save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                print(f"Saved new best model for {name} at {model_path}")
                
                # Evaluate on test set with best model
                test_loss = eval_epoch(model, test_loader, criterion, device, args.input_frames, epoch, split_name="test")

                history[name]['test_loss'].append(test_loss)
                print(f"Test Loss: {test_loss:.4f}")

                gen_mean, gen_std = eval_len_generalization(model, gen_test_loader, device, args.input_frames)

                wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
                wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
                wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})

                # ---- wandb line plot ----
                steps = np.arange(1, gen_pred_frames + 1)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(steps, gen_mean, label="MSE")
                ax.fill_between(steps,
                                (gen_mean - gen_std).clip(min=0),
                                gen_mean + gen_std,
                                alpha=0.3,
                                label="± std")
                ax.set_xlabel("Prediction horizon t")
                ax.set_ylabel("MSE")
                ax.set_title(f"Length generalization (seq_len = {args.gen_seq_len})")
                ax.legend()
                wandb.log({f"len_gen_curve": wandb.Image(fig)})
                plt.close(fig)
                
                if args.run_velocity_generalization:
                    vx, vy, err = eval_velocity_generalization(model, device, args)

                    wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                            for i, vx_i in enumerate(vx)
                            for j, vy_j in enumerate(vy)})

                    fig, ax = plt.subplots()
                    im = ax.imshow(err[::-1],  # flip y so +vy is up
                                extent=[vx[0]-0.5, vx[-1]+0.5, vy[0]-0.5, vy[-1]+0.5],
                                origin='lower')
                    ax.set_xlabel('$v_x$  (pixels / frame)')
                    ax.set_ylabel('$v_y$')
                    ax.set_title('Velocity-generalisation MSE')
                    fig.colorbar(im, ax=ax)
                    wandb.log({"vel_gen_heatmap": wandb.Image(fig)})
                    plt.close(fig)

                # Log model path and metrics to wandb
                wandb.log({
                    "best_model_path": model_path,
                    "best_val_loss": val_loss,
                    "test_loss": test_loss,
                    "epoch": epoch
                })
            
            # Log metrics to wandb
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })
        
        # Check if we've reached total_train_steps and minimum epochs
        if args.total_train_steps is not None:
            current_steps = epoch * len(train_loader)
            if current_steps >= args.total_train_steps and epoch >= args.min_epochs:
                print(f"Reached {current_steps} training steps (target: {args.total_train_steps}) and {epoch} epochs (min: {args.min_epochs}). Stopping training.")
                break

    # Print final best model paths and test results
    print("\nBest model paths and test results:")
    for name in models:
        model_filename = f"{name}_best_model_{wandb.run.id}.pth"
        model_path = os.path.join(args.model_save_dir, model_filename)
        test_loss = history[name]['test_loss'][-1] if history[name]['test_loss'] else None
        print(f"{name}: {model_path} | Test Loss: {test_loss:.4f}" if test_loss is not None else f"{name}: {model_path} | Test Loss: N/A")

if __name__ == '__main__':
    main()
