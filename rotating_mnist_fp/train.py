import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from rotating_mnist_dataset import RotatingMNISTDataset
from rotating_mnist_models import Seq2SeqStandardRNN, Seq2SeqConvRNN, Seq2SeqGalileanRNN, Seq2SeqRotationRNN
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision.utils import make_grid
import time
import sys
import os
import math
import torch.nn.functional as F

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

def train_epoch(model, dataloader, optimizer, criterion, device, input_frames, teacher_forcing_ratio):
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


def eval_len_generalization(model, dataloader, device, input_frames, max_batches=None):
    """
    Returns:
        mean_err  – numpy array [T]  (MSE at each future step, averaged over test set)
        std_err   – numpy array [T]  (sample‑wise std at each step)
        
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the test data
        device: Device to run evaluation on
        input_frames: Number of input frames to use
        max_batches: Maximum number of batches to process (None = process all)
    """
    model.eval()
    first_pass = True
    with torch.no_grad():
        n_sequences = 0
        pbar = tqdm(dataloader, desc="Evaluating Length Generalization", leave=False)
        for i, (seq, _) in enumerate(pbar):
            # Stop if we've reached the maximum number of batches
            if max_batches is not None and i >= max_batches:
                break
                
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

                log_sequence_predictions_new(inp, tgt, pred, split_name="len_gen", num_samples=10, device=device, subsample_t=2)

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




def eval_velocity_generalization(model, device, args, use_subset=True):
    """
    Returns
    -------
    v_vals  : np.ndarray [K]   (sorted angular velocities)
    err_vec : np.ndarray [K]   (mean MSE at each v)
    """
    v_vals = np.array(sorted(args.gen_vel_list))
    err_vec = np.zeros(len(v_vals))

    crit = torch.nn.MSELoss(reduction='none')
    model.eval()

    vel_pbar = tqdm(v_vals, desc="Evaluating Velocity Generalization", leave=False)
    for i, v in enumerate(vel_pbar):
        vel_pbar.set_postfix({"v": v})
        dataset = RotatingMNISTDataset(
            root=args.root,
            train=False,
            seq_len=args.seq_len,
            image_size=args.image_size,
            angular_velocities=[v],
            num_digits=1,
            random=False
        )
        
        # Use subset only if gen_vel_subset flag is True
        # if use_subset:
        #     subset, _ = torch.utils.data.random_split(
        #         dataset,
        #         [min(args.gen_vel_n_seq, len(dataset)),
        #         max(0, len(dataset) - min(args.gen_vel_n_seq, len(dataset)))],
        #         generator=torch.Generator().manual_seed(42)
        #     )
        #     loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker)
        # else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker)

        mse_sum, n_seen = 0.0, 0
        with torch.no_grad():
            batch_pbar = tqdm(loader, desc=f"v={v}", leave=False)
            for seq, _ in batch_pbar:
                seq = seq.to(device)
                inp, tgt = seq[:, :args.input_frames], seq[:, args.input_frames:]
                pred = model(inp, pred_len=tgt.size(1), teacher_forcing_ratio=0.0)
                mse = crit(pred, tgt).mean(dim=(2, 3, 4))  # [B,T]
                batch_mse = mse.mean().item()
                mse_sum += batch_mse * mse.size(0)
                n_seen += mse.size(0)
                batch_pbar.set_postfix({"mse": f"{batch_mse:.4f}"})
                if batch_pbar.n == 0:  # first batch
                    log_sequence_predictions_new(inp, tgt, pred, split_name=f"vel_gen_v{v}", num_samples=10, device=device)
                if use_subset and n_seen > 100:
                    break

        err_vec[i] = mse_sum / n_seen
    vel_pbar.close()
    return v_vals, err_vec


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
    parser.add_argument('--gen_seq_len', type=int, default=40)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--min_epochs', type=int, default=50, help='Minimum number of epochs to train')
    parser.add_argument('--total_train_steps', type=int, default=None, help='Total number of training steps (overrides epochs if not None)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', choices=['standard','conv', 'gal','rot','both'], default='both')
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--decoder_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--decoder_conv_layers', type=int, default=4)
    parser.add_argument('--group_equiv_convrnn', action='store_true', default=False)
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use (default: use all)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='Probability of using teacher forcing during training (0.0-1.0)')
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--v_list', type=int, nargs='+', default=[-40, -30, -20, -10, 0, 10, 20, 30, 40], help='List of velocities for model training')
    parser.add_argument('--data_v_list', type=int, nargs='+', default=[-40, -30, -20, -10, 0, 10, 20, 30, 40], help='List of velocities for dataset generation')
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for dataset splitting')
    parser.add_argument('--model_seed', type=int, default=None, help='Random seed for model initialization (default: random)')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the run')
    parser.add_argument('--run_len_generalization', action='store_true', help='Run length generalization test')
    parser.add_argument('--gen_vel_n_seq', type=int, default=512, help='How many sequences to sample **per (vx,vy)** for generalisation test')
    parser.add_argument('--gen_vel_list', type=int, nargs='+', default=[-40, -30, -20, -10, 0, 10, 20, 30, 40], help='Angular velocities (deg/frame) to evaluate for velocity generalization')
    parser.add_argument('--run_velocity_generalization', action='store_true', help='Run velocity generalization test')
    parser.add_argument('--model_save_dir', type=str, default='./fernn/rotmnist/', help='Directory to save model checkpoints')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model checkpoint to load for evaluation')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate the loaded model without training')
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
               name=f"rotating_mnist_fp_{args.model}_{args.run_name}")

    assert args.input_frames < args.seq_len, "input_frames must be less than seq_len"
    pred_frames = args.seq_len - args.input_frames

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and splits
    full = RotatingMNISTDataset(
        root=args.root,
        train=True,
        seq_len=args.seq_len,
        image_size=args.image_size,
        angular_velocities=args.data_v_list,
        num_digits=1
    )
    val_size = int(0.1 * len(full))
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(args.data_seed))
    
    test_dataset = RotatingMNISTDataset(
        root=args.root,
        train=False,
        seq_len=args.seq_len,
        image_size=args.image_size,
        angular_velocities=args.data_v_list,
        num_digits=1
    )

    gen_test_dataset = RotatingMNISTDataset(
            root=args.root,
            train=False,
            seq_len=args.gen_seq_len,
            image_size=args.image_size,
            angular_velocities=args.data_v_list,
            num_digits=1
    )
    gen_test_loader = DataLoader(gen_test_dataset, batch_size=args.batch_size)
    gen_pred_frames = args.gen_seq_len - args.input_frames

    # Limit training samples if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_ds):
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.data_seed))[:args.max_train_samples]
        train_ds = Subset(train_ds, indices)
        print(f"Limited training to {args.max_train_samples} samples")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_dataset)}")
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
    print(f"Velocity list: {args.v_list}")
    print(f"Data velocity list: {args.data_v_list}")
    print(f"Group equiv convrnn: {args.group_equiv_convrnn}")
    num_v = len(args.v_list)
    print(f"Num v: {num_v}")
    N_lifted = 36
    print(f"N_lifted: {N_lifted}")

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
            v_list=args.v_list,
            pool_type='max',
            decoder_conv_layers=args.decoder_conv_layers
        ).to(device)
    if args.model in ('rot','both'):
        models['rot'] = Seq2SeqRotationRNN(
            input_channels=1,
            hidden_channels=args.hidden_size,
            height=args.image_size,
            width=args.image_size,
            h_kernel_size=args.kernel_size,
            u_kernel_size=args.kernel_size,
            v_list=args.v_list,
            N=N_lifted,
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
            #     f"test_loss": test_loss,
            #     "epoch": 0
            # })

            if args.run_len_generalization:
                print("Running length generalization test...")
                gen_mean, gen_std = eval_len_generalization(model, gen_test_loader, device, args.input_frames)
                print(f"Length generalization mean MSE: {gen_mean.mean():.4f}")

                wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
                wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
                wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})


            if args.run_velocity_generalization:
                print("Running velocity generalization test...")
                v_vals, mse_vec = eval_velocity_generalization(model, device, args, use_subset=True)
                print(f"Velocity generalization results:")
                for v, mse in zip(v_vals, mse_vec):
                    print(f"Velocity {v}: MSE {mse:.4f}")
                
                wandb.log({f"vel_gen_err_v{v_i}": mse_vec[v_i] for v_i in range(len(v_vals))})

                # Line-plot (MSE vs. angular velocity)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(v_vals, mse_vec, marker="o")
                ax.set_xlabel("Angular velocity (deg / frame)")
                ax.set_ylabel("MSE")
                ax.set_title(f"{name} – velocity generalization")
                ax.grid(True)

                # Heat-map (1-D for clarity, but still useful)
                fig_hm, ax_hm = plt.subplots(figsize=(6, 1.2))
                hm = ax_hm.imshow(mse_vec[None, :], aspect="auto", cmap="viridis")
                ax_hm.set_yticks([])  # 1-D heat-map – hide y-axis
                ax_hm.set_xticks(np.arange(len(v_vals)))
                ax_hm.set_xticklabels(v_vals, rotation=45, ha="right")
                ax_hm.set_title("Velocity-generalization heat-map")
                plt.colorbar(hm, ax=ax_hm, shrink=0.6, label="MSE")

                wandb.log({
                    f"vel_gen_curve"  : wandb.Image(fig),
                    f"vel_gen_heatmap": wandb.Image(fig_hm),
                    f"vel_gen_raw"    : wandb.Table(
                        data=list(zip(v_vals, mse_vec)),
                        columns=["velocity_deg_per_frame", "mse"]
                    )
                })

                plt.close(fig)
                plt.close(fig_hm)


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
            train_loss = train_epoch(model, train_loader, optimizers[name], criterion, device, args.input_frames, args.teacher_forcing_ratio)
            val_loss = eval_epoch(model, val_loader, criterion, device, args.input_frames, epoch, split_name="val")
            
            # Check if loss is NaN or infinity
            if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)) or \
               torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
                print(f"Exiting due to NaN or Inf loss: Train Loss: {train_loss}, Val Loss: {val_loss}")
                sys.exit(1)
                
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

                if args.run_len_generalization:
                    max_batches = None if epoch % 10 == 0 else 10
                    gen_mean, gen_std = eval_len_generalization(model, gen_test_loader, device, args.input_frames, max_batches=max_batches)

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
                                    label="± std")
                    ax.set_xlabel("Prediction horizon t")
                    ax.set_ylabel("MSE")
                    ax.set_title(f"Length generalization (seq_len = {args.gen_seq_len})")
                    ax.legend()
                    wandb.log({f"len_gen_curve": wandb.Image(fig)})
                    plt.close(fig)
                
                if args.run_velocity_generalization:
                    use_subset = False if epoch % 10 == 0 else True
                    v_vals, mse_vec = eval_velocity_generalization(model, device, args, use_subset=use_subset)

                    wandb.log({f"vel_gen_err_v{v_i}": mse_vec[v_i] for v_i in range(len(v_vals))})

                    # Line-plot (MSE vs. angular velocity)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(v_vals, mse_vec, marker="o")
                    ax.set_xlabel("Angular velocity (deg / frame)")
                    ax.set_ylabel("MSE")
                    ax.set_title(f"{name} – velocity generalization")
                    ax.grid(True)

                    # Heat-map (1-D for clarity, but still useful)
                    fig_hm, ax_hm = plt.subplots(figsize=(6, 1.2))
                    hm = ax_hm.imshow(mse_vec[None, :], aspect="auto", cmap="viridis")
                    ax_hm.set_yticks([])  # 1-D heat-map – hide y-axis
                    ax_hm.set_xticks(np.arange(len(v_vals)))
                    ax_hm.set_xticklabels(v_vals, rotation=45, ha="right")
                    ax_hm.set_title("Velocity-generalization heat-map")
                    plt.colorbar(hm, ax=ax_hm, shrink=0.6, label="MSE")

                    wandb.log({
                        f"vel_gen_curve"  : wandb.Image(fig),
                        f"vel_gen_heatmap": wandb.Image(fig_hm),
                        f"vel_gen_raw"    : wandb.Table(
                            data=list(zip(v_vals, mse_vec)),
                            columns=["velocity_deg_per_frame", "mse"]
                        )
                    })

                    plt.close(fig)
                    plt.close(fig_hm)


                # Log model path to wandb
                wandb.log({
                    f"best_model_path": model_path,
                    f"best_val_loss": val_loss,
                    f"test_loss": test_loss, # Log test_loss for this model
                    "epoch": epoch
                })
            
            # Log metrics to wandb
            wandb.log({
                f"train_loss": train_loss,
                f"val_loss": val_loss,
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
        test_loss = history[name]['test_loss'][-1] if history[name]['test_loss'] else None # Get the last recorded test loss
        print(f"{name}: {model_path} | Test Loss: {test_loss:.4f}" if test_loss is not None else f"{name}: {model_path} | Test Loss: N/A")

if __name__ == '__main__':
    main()
