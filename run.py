from training import train_worker

# Other imports
import argparse
import torch
import numpy as np
import random
import os
import wandb
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument("--checkpoint_path", type=str, help="Path to current model", default="checkpoint.pt")
    parser.add_argument("--best_model_path", type=str, help="Path to best model", default="best_model.pt")
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
    parser.add_argument("--run_name", type=str, default="Null", help="Run name for wandb")
    parser.add_argument("--wandb_api_key", type=str, required=True, help="Wandb API key")
    # New arguments for loading best model from W&B
    parser.add_argument("--load_best_model", action="store_true", help="Load the best model from W&B")
    parser.add_argument("--wandb_run_path", type=str, default=None, 
                        help="W&B run path in format: <username>/<project-name>/<run-id>")
    parser.add_argument("--artifact_name", type=str, default="best_model:latest",
                        help="Name of the artifact to load")
    
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
def download_best_model_from_wandb(args):
    """
    Download the best model from W&B artifacts and save it to best_model_path
    
    Args:
        args: Command line arguments
        
    Returns:
        success: Boolean indicating whether the download was successful
    """
    try:
        # Initialize W&B API
        api = wandb.Api()
        
        # Get the specified run
        run = api.run(args.wandb_run_path)
        
        # Get the best model artifact from the run
        # Extract the entity and project from run path
        path_parts = args.wandb_run_path.split('/')
        if len(path_parts) != 3:
            raise ValueError(f"Invalid wandb_run_path format: {args.wandb_run_path}. Expected format: 'entity/project/run_id'")
            
        entity, project, run_id = path_parts
        
        # Get a list of artifacts logged to this run
        print(f"Fetching artifacts from run: {args.wandb_run_path}")
        run_artifacts = run.logged_artifacts()
        if not run_artifacts:
            print("No artifacts found in this run. Available runs in project:")
            runs = api.runs(f"{entity}/{project}")
            for r in runs:
                print(f"Run ID: {r.id}, Name: {r.name}")
            raise ValueError(f"No artifacts found in run {run_id}")
        
        # Print available artifacts
        print("Available artifacts in this run:")
        for art in run_artifacts:
            print(f"- {art.name}:{art.version}")
        
        # Use the first model artifact found or the specified one if it exists
        model_artifacts = [art for art in run_artifacts if "model" in art.name.lower()]
        
        if not model_artifacts:
            raise ValueError(f"No model artifacts found in run {run_id}")
        
        # Use specified artifact if it exists, otherwise use the first model artifact
        artifact_name = args.artifact_name.split(':')[0]  # Remove version if specified
        matching_artifacts = [art for art in model_artifacts if artifact_name in art.name]
        
        if matching_artifacts:
            artifact = matching_artifacts[0]
        else:
            print(f"Specified artifact '{args.artifact_name}' not found. Using '{model_artifacts[0].name}:{model_artifacts[0].version}' instead.")
            artifact = model_artifacts[0]
            
        print(f"Downloading artifact: {artifact.name}:{artifact.version}")
        artifact_dir = artifact.download()
        
        # Find the model file in the artifact directory
        model_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        if not model_files:
            raise ValueError(f"No model files found in artifact directory {artifact_dir}")
        
        # Path to the first model file found
        source_path = os.path.join(artifact_dir, model_files[0])
        print(f"Found model at {source_path}")
        
        # Copy the model file to the best_model_path
        shutil.copy(source_path, args.best_model_path)
        print(f"Successfully saved best model from W&B to {args.best_model_path}")
        
        return True
    
    except Exception as e:
        print(f"Error downloading model from W&B: {e}")
        return False
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
    
    # Download best model from W&B if requested
    if args.load_best_model and args.wandb_run_path:
        print(f"Downloading best model from W&B run: {args.wandb_run_path}")
        success = download_best_model_from_wandb(args)
        
        if success:
            print(f"Successfully downloaded best model from W&B to {args.best_model_path}")
        else:
            print("Failed to download model from W&B")
    
    # Train model
    model, train_state = train_worker(
        device=device,
        config=config,
        resume_training=args.resume_training, 
        checkpoint_path=args.checkpoint_path,
        best_model_path=args.best_model_path, 
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        run_name=args.run_name
    )
    
    print("Training completed!")
    

if __name__ == "__main__":
    main()