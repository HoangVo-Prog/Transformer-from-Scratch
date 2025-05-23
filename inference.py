import torch
import argparse
import os
from model import Transformer
from utils import subsequent_mask
from Data.data import cache_or_process
from config import pad_token


class TranslationInference:
    def __init__(self, model_path, device=None):
        """
        Initialize the translation inference object
        
        Args:
            model_path (str): Path to the saved model
            device (torch.device): Device to run the model on
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load tokenizers
        _, _, _, self.en_tokenizer, self.vi_tokenizer = cache_or_process()
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Get special tokens
        self.pad_idx = self.en_tokenizer.decode(pad_token).ids
        self.start_symbol = self.vi_tokenizer.token_to_id("[BOS]")
        self.end_symbol = self.vi_tokenizer.token_to_id("[EOS]")
    
    def _load_model(self, model_path):
        """
        Load the model from the given path
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            Transformer: Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Create model with same architecture as training
        model = Transformer(
            N=6, 
            d_model=512, 
            d_ff=2048, 
            h=8, 
            dropout=0.1,
        )
        
        # Load the saved model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Model loaded from {model_path}")
        return model
    
    def translate_sentence(self, sentence, max_len=100, beam_size=None):
        """
        Translate a single sentence from English to Vietnamese
        
        Args:
            sentence (str): English sentence to translate
            max_len (int): Maximum length of the generated translation
            beam_size (int, optional): Size for beam search. If None, uses greedy decoding
            
        Returns:
            str: Translated Vietnamese sentence
        """
        self.model.eval()
        
        # Tokenize the input sentence
        token_ids = self.en_tokenizer.encode(sentence).ids
        src = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        src_mask = (src != self.pad_idx).unsqueeze(-2)
        
        if beam_size is not None and beam_size > 1:
            output = self._beam_search(src, src_mask, max_len, beam_size)
        else:
            output = self._greedy_decode(src, src_mask, max_len)
        
        # Convert token IDs to text
        translated_tokens = output[0].cpu().numpy()
        
        # Find EOS token
        if self.end_symbol in translated_tokens:
            eos_idx = list(translated_tokens).index(self.end_symbol)
            translated_tokens = translated_tokens[:eos_idx]
            
        translated_text = self.vi_tokenizer.decode(translated_tokens)
        return translated_text
    
    def _greedy_decode(self, src, src_mask, max_len):
        """
        Perform greedy decoding
        
        Args:
            src (torch.Tensor): Source tensor
            src_mask (torch.Tensor): Source mask
            max_len (int): Maximum length of the generated translation
            
        Returns:
            torch.Tensor: Generated sequence
        """
        with torch.no_grad():
            memory = self.model.encode(src, src_mask)
            ys = torch.zeros(1, 1).fill_(self.start_symbol).type_as(src.data).to(self.device)
            
            for i in range(max_len - 1):
                # Generate next token
                tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(self.device)
                out = self.model.decode(memory, src_mask, ys, tgt_mask)
                prob = self.model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()
                
                # Add next token to output
                ys = torch.cat(
                    [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word).to(self.device)], 
                    dim=1
                )
                
                # Stop if end of sentence
                if next_word == self.end_symbol:
                    break
                    
            return ys
            
    def _beam_search(self, src, src_mask, max_len, beam_size):
        """
        Perform beam search decoding
        
        Args:
            src (torch.Tensor): Source tensor
            src_mask (torch.Tensor): Source mask
            max_len (int): Maximum length of the generated translation
            beam_size (int): Beam size
            
        Returns:
            torch.Tensor: Generated sequence
        """
        with torch.no_grad():
            # Encode the source sequence
            memory = self.model.encode(src, src_mask)
            
            # Initialize the beam
            ys = torch.zeros(beam_size, 1).fill_(self.start_symbol).type_as(src.data).to(self.device)
            log_probs = torch.zeros(beam_size).to(self.device)
            
            finished_beams = []
            finished_scores = []
            
            # Beam search
            for i in range(max_len - 1):
                # For each beam
                candidates = []
                
                for j in range(ys.size(0)):
                    # Skip if the beam is finished
                    if ys[j, -1].item() == self.end_symbol:
                        candidates.append((log_probs[j], ys[j]))
                        continue
                    
                    # Generate next token probabilities
                    tgt_mask = subsequent_mask(ys[j].size(0)).type_as(src.data).to(self.device)
                    out = self.model.decode(memory, src_mask, ys[j].unsqueeze(0), tgt_mask)
                    prob = self.model.generator(out[:, -1])
                    log_prob = torch.log_softmax(prob, dim=-1)
                    
                    # Get top k candidates
                    topk_prob, topk_idx = torch.topk(log_prob, beam_size)
                    
                    for k in range(beam_size):
                        score = log_probs[j] + topk_prob[0, k].item()
                        token = topk_idx[0, k].item()
                        
                        new_beam = torch.cat(
                            [ys[j], torch.ones(1).type_as(src.data).fill_(token).to(self.device)], 
                            dim=0
                        )
                        
                        candidates.append((score, new_beam))
                        
                        # If EOS, add to finished beams
                        if token == self.end_symbol:
                            finished_beams.append(new_beam)
                            finished_scores.append(score)
                
                # Sort and keep top beam_size candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:beam_size]
                
                # Update beams
                ys = torch.zeros(beam_size, i+2).type_as(src.data).to(self.device)
                log_probs = torch.zeros(beam_size).to(self.device)
                
                for j in range(min(beam_size, len(candidates))):
                    log_probs[j] = candidates[j][0]
                    ys[j, :] = candidates[j][1]
                
                # If all beams are finished or we reach max_len
                if all(ys[:, -1].eq(self.end_symbol)) or i == max_len - 2:
                    break
            
            # Find the best translation
            if finished_beams:
                # Choose the best finished beam
                best_idx = finished_scores.index(max(finished_scores))
                best_beam = finished_beams[best_idx].unsqueeze(0)
                return best_beam
            else:
                # Choose the best current beam
                best_idx = log_probs.argmax().item()
                return ys[best_idx].unsqueeze(0)
    
    def translate_file(self, input_file, output_file, beam_size=None):
        """
        Translate all sentences in a file
        
        Args:
            input_file (str): Path to the input file (English)
            output_file (str): Path to the output file (Vietnamese)
            beam_size (int, optional): Size for beam search
        """
        with open(input_file, 'r', encoding='utf-8') as f_in:
            sentences = [line.strip() for line in f_in]
        
        translations = []
        total = len(sentences)
        
        for i, sentence in enumerate(sentences):
            print(f"Translating sentence {i+1}/{total}", end='\r')
            translation = self.translate_sentence(sentence, beam_size=beam_size)
            translations.append(translation)
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for translation in translations:
                f_out.write(translation + '\n')
        
        print(f"\nTranslated {total} sentences and saved to {output_file}")
    
    def interactive_translation(self):
        """
        Interactive translation mode
        """
        print("=== Interactive Translation Mode (English to Vietnamese) ===")
        print("Enter 'q' or 'quit' to exit")
        
        while True:
            user_input = input("\nEnglish: ")
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
                
            if not user_input.strip():
                continue
                
            translation = self.translate_sentence(user_input)
            print(f"Vietnamese: {translation}")


def main():
    parser = argparse.ArgumentParser(description="Transformer Translation Inference")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", 
                        help="Path to the saved model")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input file (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output file (optional)")
    parser.add_argument("--beam_size", type=int, default=None,
                        help="Beam size for beam search decoding (default: greedy)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (default: cuda if available, else cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the translator
    translator = TranslationInference(args.model, device)
    
    # Translate file or start interactive mode
    if args.input and args.output:
        translator.translate_file(args.input, args.output, beam_size=args.beam_size)
    else:
        translator.interactive_translation()


if __name__ == "__main__":
    main()
