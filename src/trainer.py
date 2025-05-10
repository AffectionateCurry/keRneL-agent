import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, create_reference_model
from datasets import Dataset
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging
from tqdm import tqdm

from kernelbench_grpo_env import KernelBenchGRPOEnv


logger = logging.getLogger(__name__)


class KernelBenchGRPOTrainer:
    """
    GRPO trainer for optimizing CUDA kernels using KernelBench.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",  # Small Qwen model
        gpt_model: str = "gpt-4o-2024-08-06",
        level: int = 1,
        device: str = "cuda",
        output_dir: str = "./grpo_output",
        logging_dir: str = "./logs",
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-6,
        num_epochs: int = 3,
        max_steps: int = 1000,
        save_steps: int = 100,
    ):
        """Initialize the GRPO trainer."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Create reference model for GRPO
        self.ref_model = create_reference_model(self.model)
        
        # Initialize environment
        self.env = KernelBenchGRPOEnv(
            level=level,
            gpt_model=gpt_model,
            device=self.device
        )
        
        # GRPO configuration
        self.config = GRPOConfig(
            output_dir=str(self.output_dir),
            logging_dir=logging_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            save_steps=save_steps,
            logging_steps=10,
            eval_steps=50,
            warmup_steps=100,
            fp16=device == "cuda",
            remove_unused_columns=False,
            reward_model_type="custom",  # We calculate our own rewards
        )
        
    def collect_trajectories(
        self, 
        num_trajectories: int = 100,
        problems_per_trajectory: int = 1,
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect trajectories for training.
        
        Args:
            num_trajectories: Number of trajectories to collect
            problems_per_trajectory: Number of problems per trajectory
            save_path: Optional path to save trajectories
            
        Returns:
            List of trajectory dictionaries
        """
        logger.info(f"Collecting {num_trajectories} trajectories...")
        trajectories = []
        
        for i in tqdm(range(num_trajectories), desc="Collecting trajectories"):
            for _ in range(problems_per_trajectory):
                try:
                    trajectory = self.env.generate_trajectory(
                        self.model, 
                        self.tokenizer
                    )
                    trajectories.append(trajectory)
                except Exception as e:
                    logger.error(f"Error collecting trajectory: {e}")
                    continue
        
        # Save trajectories if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(trajectories, f)
        
        return trajectories
    
    def prepare_dataset(self, trajectories: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare trajectories for GRPO training.
        
        Args:
            trajectories: List of collected trajectories
            
        Returns:
            Hugging Face Dataset for training
        """
        # Flatten trajectories into individual transitions
        data = []
        
        for trajectory in trajectories:
            observations = trajectory["observations"]
            actions = trajectory["actions"]
            rewards = trajectory["rewards"]
            
            for i in range(len(actions)):
                data.append({
                    "input_ids": self.tokenizer(
                        observations[i], 
                        truncation=True, 
                        max_length=2048,
                        return_tensors="pt"
                    )["input_ids"].squeeze(0),
                    "labels": self.tokenizer(
                        actions[i], 
                        truncation=True, 
                        max_length=256,
                        return_tensors="pt"
                    )["input_ids"].squeeze(0),
                    "reward": rewards[i],
                    "observation": observations[i],
                    "action": actions[i]
                })
        
        return Dataset.from_list(data)
    
    def compute_rewards(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Custom reward computation for GRPO.
        
        Args:
            batch: Batch of data from the dataset
            
        Returns:
            Tensor of rewards
        """
        # The rewards are already computed during trajectory collection
        return torch.tensor(batch["reward"], dtype=torch.float32)
    
    def train(
        self, 
        trajectories: Optional[List[Dict[str, Any]]] = None,
        num_trajectories: int = 100
    ):
        """
        Train the model using GRPO.
        
        Args:
            trajectories: Pre-collected trajectories (optional)
            num_trajectories: Number of trajectories to collect if not provided
        """
        # Collect trajectories if not provided
        if trajectories is None:
            trajectories = self.collect_trajectories(num_trajectories)
        
        # Prepare dataset
        dataset = self.prepare_dataset(trajectories)
        
        # Initialize GRPO trainer
        trainer = GRPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            config=self.config,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            compute_rewards=self.compute_rewards,
        )
        
        # Train
        logger.info("Starting GRPO training...")
        trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(final_model_path)
        logger.info(f"Training complete. Model saved to {final_model_path}")
    
    def evaluate(self, num_problems: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained model on a set of problems.
        
        Args:
            num_problems: Number of problems to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {num_problems} problems...")
        
        results = {
            "total_problems": num_problems,
            "average_speedup": 0.0,
            "best_speedup": 0.0,
            "correct_solutions": 0,
            "problems_with_speedup": 0
        }
        
        speedups = []
        
        for i in range(num_problems):
            try:
                trajectory = self.env.generate_trajectory(
                    self.model, 
                    self.tokenizer
                )
                
                final_results = self.env.get_final_results()
                best_speedup = final_results["best_speedup"]
                
                speedups.append(best_speedup)
                
                # Count metrics
                if best_speedup > 0:
                    results["correct_solutions"] += 1
                if best_speedup > 1.0:
                    results["problems_with_speedup"] += 1
                    
            except Exception as e:
                logger.error(f"Error evaluating problem {i}: {e}")
                speedups.append(0.0)
        
        # Calculate averages
        results["average_speedup"] = sum(speedups) / len(speedups) if speedups else 0.0
        results["best_speedup"] = max(speedups) if speedups else 0.0
        
        logger.info(f"Evaluation results: {results}")
        return results


# Example usage script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Qwen model for kernel optimization using GRPO")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--level", type=int, default=1, help="KernelBench difficulty level")
    parser.add_argument("--num_trajectories", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    
    # Initialize trainer
    trainer = KernelBenchGRPOTrainer(
        model_name=args.model_name,
        level=args.level,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )
    
    if args.evaluate_only:
        # Load pre-trained model if evaluating
        model_path = Path(args.output_dir) / "final_model"
        if model_path.exists():
            trainer.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Evaluate
        results = trainer.evaluate(num_problems=10)
        print(json.dumps(results, indent=2))
    else:
        # Train
        trainer.train(num_trajectories=args.num_trajectories)
        
        # Evaluate after training
        results = trainer.evaluate(num_problems=10)
        
        # Save evaluation results
        with open(Path(args.output_dir) / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)