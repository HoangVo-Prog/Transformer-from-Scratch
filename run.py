from training import train_worker

# Other imports
import argparse
import torch
import numpy as np
import random
import os
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument("--checkpoint_path", type=str, help="Path to current model", default="")
    parser.add_argument("--best_model_path", type=str, help="Path to best model", default="")
    parser.add_argument("--resume_training", type=bool, help="Define whether train from scratch or not", default=False)
    parser.add_argument("--n_epochs", type=int, required=True, help="Path to number of epochs")
    parser.add_argument("--base_lr", type=float, help="Define learning rate", default=1e-4)
    parser.add_argument("--warmup", type=int, help="Define warmup step", default=4000)
    parser.add_argument("--accum_iter", type=int, help="Define gradient accumulation steps", default=4)
    parser.add_argument("--N", type=int, help="Define number of layers", default=6)
    parser.add_argument("--d_model", type=int, help="Define number of model dimension", default=512)
    parser.add_argument("--d_ff", type=int, help="Define number of feed forward dimension", default=2048)
    parser.add_argument("--h", type=int, help="Define number of heads", default=8)
    parser.add_argument("--dropout", type=float, help="Define dropout rate", default=0.1)
    parser.add_argument("--seed", type=int, help="Define random seed", default=42)
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--project_name", type=str, default="transformer-translation",
                      help="Project name for wandb")
    parser.add_argument("--entity", type=str, default="Vo Hoang", help="Entity name for wandb")
    parser.add_argument("--run_name", type=str, default="Null", help="Run name for wandb")
    parser.add_argument("--wandb_api_key", type=str, required=True, help="Wandb API key")
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    np.random.seed(seed)
    random.seed(seed)

    # Ensures that CUDA selects deterministic algorithms (if available)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    wandb.login(key=args.wandb_api_key)
    print("Wandb login successful!")
    
    config = {
        "base_lr": args.base_lr, 
        "warmup": args.warmup, 
        "n_epochs": args.n_epochs, 
        "accum_iter": args.accum_iter, 
        "N": args.N,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "h": args.h,
        "dropout": args.dropout
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model
    model, train_state = train_worker(
        device=device,
        config=config,
        resume_training=args.resume_training, 
        checkpoint_path=args.checkpoint_path,
        best_model_path=args.best_model_path, 
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        entity=args.entity,
        run_name=args.run_name
    )
    
    print("Training completed!")
    


if __name__ == "__main__":
    main()