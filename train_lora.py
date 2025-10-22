"""
LoRA Fine-tuning Script for Face Models
Trains personalized face LoRAs based on reference images
"""

import os
import torch
from pathlib import Path
from PIL import Image
from typing import Optional, List
import yaml
from omegaconf import OmegaConf
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import argparse


class FaceDataset(Dataset):
    """Dataset for face images with captions"""
    
    def __init__(
        self,
        image_dir: str,
        trigger_word: str,
        resolution: int = 512,
        caption_file: Optional[str] = None,
    ):
        self.image_dir = Path(image_dir)
        self.trigger_word = trigger_word
        self.resolution = resolution
        
        # Get all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(self.image_dir.glob(ext))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images for training")
        
        # Load or generate captions
        self.captions = self._load_captions(caption_file)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def _load_captions(self, caption_file: Optional[str]) -> List[str]:
        """Load or generate captions for training"""
        if caption_file and Path(caption_file).exists():
            with open(caption_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # Generate default captions with trigger word
        captions = [
            f"portrait photo of {self.trigger_word}",
            f"close-up photo of {self.trigger_word}",
            f"{self.trigger_word} face",
            f"headshot of {self.trigger_word}",
            f"professional photo of {self.trigger_word}",
        ]
        
        # Repeat to match number of images
        return (captions * (len(self.image_paths) // len(captions) + 1))[:len(self.image_paths)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        caption = self.captions[idx]
        
        return {
            'pixel_values': image,
            'input_ids': caption,
        }


def collate_fn(examples, tokenizer, text_encoder):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    captions = [example['input_ids'] for example in examples]
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    return {
        'pixel_values': pixel_values,
        'input_ids': inputs.input_ids,
    }


def train_lora_model(
    face_name: str,
    image_dir: str,
    trigger_word: str,
    output_dir: str,
    config: dict,
):
    """Train a LoRA model for a specific face"""
    
    print(f"\n{'='*60}")
    print(f"Training LoRA for: {face_name}")
    print(f"Trigger word: {trigger_word}")
    print(f"Images directory: {image_dir}")
    print(f"{'='*60}\n")
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        mixed_precision=config.get('mixed_precision', 'no'),
    )
    
    # Load base model
    print("Loading base model...")
    base_model = config['base_model']
    
    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder='tokenizer',
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_model,
        subfolder='text_encoder',
    )
    vae = AutoencoderKL.from_pretrained(
        base_model,
        subfolder='vae',
    )
    unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder='unet',
    )
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config['rank'],
        lora_alpha=config['lora_alpha'],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if config.get('gradient_checkpointing', False):
        unet.enable_gradient_checkpointing()
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = FaceDataset(
        image_dir=image_dir,
        trigger_word=trigger_word,
        resolution=config['resolution'],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, text_encoder),
        num_workers=2,
    )
    
    # Setup optimizer
    if config.get('use_8bit_adam', False):
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                unet.parameters(),
                lr=config['learning_rate'],
            )
        except ImportError:
            print("bitsandbytes not available, using standard AdamW")
            optimizer = torch.optim.AdamW(
                unet.parameters(),
                lr=config['learning_rate'],
            )
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=config['learning_rate'],
        )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_model,
        subfolder='scheduler',
    )
    
    # Prepare with accelerator
    unet, optimizer, dataloader = accelerator.prepare(
        unet, optimizer, dataloader
    )
    
    # Move models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    progress_bar = tqdm(
        range(config['max_train_steps']),
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(config['max_train_steps'] // len(dataloader) + 1):
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch['pixel_values'].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch['input_ids'].to(accelerator.device))[0]
                
                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backprop
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            global_step += 1
            
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({"loss": loss.detach().item()})
            
            # Save checkpoint
            if global_step % config['checkpointing_steps'] == 0:
                if accelerator.is_local_main_process:
                    save_path = Path(output_dir) / f"{face_name}_step_{global_step}.safetensors"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    unet.save_pretrained(save_path.parent, safe_serialization=True)
                    print(f"\nSaved checkpoint to {save_path}")
            
            if global_step >= config['max_train_steps']:
                break
        
        if global_step >= config['max_train_steps']:
            break
    
    # Save final model
    if accelerator.is_local_main_process:
        final_path = Path(output_dir) / f"{face_name}_face.safetensors"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        unet.save_pretrained(final_path.parent, safe_serialization=True)
        print(f"\n{'='*60}")
        print(f"Training complete! Model saved to: {final_path}")
        print(f"{'='*60}\n")
    
    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description="Train face LoRA models")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--face', type=str, help='Specific face to train (optional, trains all if not specified)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config['training']
    
    # Train specific face or all faces
    faces_to_train = {}
    if args.face:
        if args.face in config['faces']:
            faces_to_train[args.face] = config['faces'][args.face]
        else:
            raise ValueError(f"Face '{args.face}' not found in config")
    else:
        faces_to_train = config['faces']
    
    # Train each face
    for face_key, face_config in faces_to_train.items():
        train_lora_model(
            face_name=face_config['name'],
            image_dir=face_config['reference_images'],
            trigger_word=face_config['trigger_word'],
            output_dir=training_config['output_dir'],
            config=training_config,
        )


if __name__ == '__main__':
    main()
