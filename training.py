from utils import subsequent_mask
from lr_scheduler import rate
from regularization import LabelSmoothing
from model import make_model
from Data.data import cache_or_process
from config import pad_token, sos_token, eos_token
from helper_function import DummyOptimizer, DummyScheduler

# Other imports
import time
import torch
from torch.optim.lr_scheduler import LambdaLR
import os
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import wandb


# Batches and Masking
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # pad=2 is the default padding index
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    
    
# Training Loop
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    def __init__(self):
        self.step = 0  # Steps in the current epoch
        self.accum_step = 0  # Number of gradient accumulation steps
        self.samples = 0  # total # of examples used
        self.tokens = 0  # total # of tokens processed
        
        # For tracking metrics
        self.train_losses = []
        self.valid_losses = []
        self.bleu_scores = []
        self.epoch = 0
    
    
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=None,
):
    """Train a single epoch"""
    if train_state is None:
        train_state = TrainState()
        
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


# Loss computation
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


# Greedy Decoding
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# Function to compute BLEU score
def compute_bleu(model, data_iter, en_tokenizer, vi_tokenizer, device, max_len=100):
    model.eval()
    references = []
    hypotheses = []
    start_symbol = vi_tokenizer.token_to_id(sos_token)
    
    with torch.no_grad():
        for batch in data_iter:
            src = batch["src_ids"].to(device)
            tgt = batch["trg_ids"].to(device)
            
            # Get reference translations
            tgt_texts = [vi_tokenizer.decode(ids) for ids in tgt.cpu().numpy()]
            references.extend([[ref.split()] for ref in tgt_texts])
            
            # Generate translations
            for i in range(src.size(0)):
                src_i = src[i:i+1]
                src_mask = (src_i != en_tokenizer.encode(pad_token).ids[0]).unsqueeze(-2)
                pred = greedy_decode(model, src_i, src_mask, max_len, start_symbol)
                pred_text = vi_tokenizer.decode(pred[0].cpu().numpy())
                hypotheses.append(pred_text.split())
    
    # Calculate BLEU score
    bleu = corpus_bleu(references, hypotheses)
    return bleu


def save_checkpoint(model, optimizer, scheduler, train_state, filename="checkpoint.pt"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_state': train_state,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")
      
    
def load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pt"):
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return None
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_state = checkpoint['train_state']
    print(f"Checkpoint loaded from {filename}")
    return train_state


# Function to plot training metrics
def plot_training_progress(train_state, save_path="training_progress.png"):
    plt.figure(figsize=(12, 8))
    
    # Convert tensors to CPU and then to numpy arrays
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        elif isinstance(tensor, list):
            # Check if list contains tensors and convert them
            if len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
                return np.array([t.cpu().detach().item() for t in tensor])
            return np.array(tensor)
        return tensor
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    train_losses = to_numpy(train_state.train_losses)
    valid_losses = to_numpy(train_state.valid_losses)
    
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot BLEU score
    plt.subplot(2, 1, 2)
    bleu_scores = to_numpy(train_state.bleu_scores)
    
    plt.plot(bleu_scores, label='BLEU Score', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    plt.title('Validation BLEU Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training progress plot saved to {save_path}")


def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)
        

def train_worker(
    device,
    config,
    resume_training=False,
    checkpoint_path="checkpoint.pt",
    best_model_path="best_model.pt",
    use_wandb=True,
    project_name="transformer-translation",
    run_name=None
):
    print(f"Train worker process using GPU: {device} for training")
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path) if os.path.dirname(best_model_path) else '.', exist_ok=True)

    # Initialize wandb
    if use_wandb:
        if run_name is None:
            run_name = f"transformer-run-{time.strftime('%Y%m%d-%H%M%S')}"
        
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "architecture": "Transformer",
                "dataset": "IWSLT2015-en-vi",
                "learning_rate": config["base_lr"],
                "warmup_steps": config["warmup"],
                "epochs": config["n_epochs"],
                "accumulation_steps": config["accum_iter"],
                "d_model": config["d_model"],
                "N": config["N"],
                "d_ff": config["d_ff"],
                "h": config["h"],
                "dropout": config["dropout"],
                "seed": 42
            }
        )
        
        # Log code files (optional)
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    train_dataloader, valid_dataloader, _, en_tokenizer, vi_tokenizer = cache_or_process()

    model = make_model(
        en_tokenizer.get_vocab_size(),
        vi_tokenizer.get_vocab_size(),  
        N=config["N"], 
        d_model=config["d_model"], 
        d_ff=config["d_ff"], 
        h=config["h"], 
        dropout=config["dropout"],
    )
    model.to(device)
    module = model
    

    pad_idx = en_tokenizer.encode(pad_token).ids[0]

    criterion = LabelSmoothing(
        size=vi_tokenizer.get_vocab_size(), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.to(device)
    

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, config["d_model"], factor=1, warmup=config["warmup"]
        ),
    )
    
    train_state = TrainState()
    best_bleu = 0.0

    if resume_training:
        loaded_state = load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path)
        if loaded_state:
            train_state = loaded_state
            print(f"Resuming from epoch {train_state.epoch}")
            if use_wandb:
                # If resuming training, log that we're continuing from a checkpoint
                wandb.log({"resumed_from_epoch": train_state.epoch})
                
    for epoch in range(train_state.epoch, config["n_epochs"]):
        train_state.epoch = epoch
        print(f"\nEpoch {epoch+1}/{config['n_epochs']}")

        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss, train_state = run_epoch(
            (Batch(b["src_ids"], b["trg_ids"], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )
        train_state.train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        valid_loss, _ = run_epoch(
            (Batch(b["src_ids"], b["trg_ids"], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        train_state.valid_losses.append(valid_loss)
        print(f"Validation Loss: {valid_loss:.4f}")
        
        
        # Compute BLEU score
        bleu = compute_bleu(model, valid_dataloader, en_tokenizer, vi_tokenizer, device)
        train_state.bleu_scores.append(bleu)
        print(f"Validation BLEU Score: {bleu:.4f}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics to wandb
        if use_wandb:
            lr = optimizer.param_groups[0]["lr"]
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "bleu_score": bleu,
                "learning_rate": lr,
                "epoch_time_seconds": epoch_time,
                "tokens_processed": train_state.tokens,
                "steps": train_state.step,
            })
        
        
        # Save regular checkpoint
        save_checkpoint(model, optimizer, lr_scheduler, train_state, checkpoint_path)
        
        # Save best model based on BLEU score
        if bleu > best_bleu:
            best_bleu = bleu
            save_checkpoint(model, optimizer, lr_scheduler, train_state, best_model_path)
            print(f"New best model saved with BLEU score: {best_bleu:.4f}")
            
            if use_wandb:
                # Log best model to wandb
                wandb.log({"best_bleu": best_bleu, "best_epoch": epoch + 1})
                
                # Save best model to wandb (optional)
                wandb.save(best_model_path)
            
        torch.cuda.empty_cache()
        
        

    # Plot and save training progress
    plot_training_progress(train_state)
    
    if use_wandb:
        # Log the final plot to wandb
        wandb.log({"training_progress_plot": wandb.Image("training_progress.png")})
        
        # Log final model to wandb artifacts
        model_artifact = wandb.Artifact(
            name=f"transformer-model-{wandb.run.id}", 
            type="model",
            description="Trained Transformer translation model"
        )
        model_artifact.add_file(best_model_path)
        wandb.log_artifact(model_artifact)
        
        # Finish the wandb run
        wandb.finish()
    
    return model, train_state    
 