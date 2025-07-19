import os
import wandb
from huggingface_hub import login

# Import configurations
import config

# Import modules from src folder
from src.data_loader import KaggleDataLoader
from src.preprocessor import DataPreprocessor
from src.dataset_builder import TextDatasetBuilder
from src.model_builder import ModelBuilder
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator

def setup_logins():
    """Handles logins for Hugging Face and W&B using environment variables."""
    print("--- Setting up logins ---")
    try:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        wandb_key = os.environ.get("WANDB_API_KEY")

        if hf_token:
            login(token=hf_token)
            print("Successfully logged into Hugging Face Hub.")
        else:
            print("WARNING: Hugging Face token not found. Set the HUGGINGFACE_TOKEN environment variable.")

        if wandb_key:
            wandb.login(key=wandb_key)
            print("Successfully logged into Weights & Biases.")
        else:
            print("WARNING: W&B API key not found. Set the WANDB_API_KEY environment variable.")
    except Exception as e:
        print(f"An error occurred during login: {e}")


def run_pipeline():
    """Main function to run the complete ML pipeline."""
    setup_logins()

    # --- Step 1: Load and Preprocess Data ---
    # This step is done only once for all models.
    data_loader = KaggleDataLoader(config.DATASET_ID, config.TRUE_FILENAME, config.FAKE_FILENAME)
    df = data_loader.load()
    
    preprocessor = DataPreprocessor(df)
    processed_df = preprocessor.preprocess()

    # --- Step 2: Loop through models, train, and evaluate ---
    for model_name in config.MODELS_TO_RUN:
        print(f"\n{'='*25} RUNNING PIPELINE FOR: {model_name.upper()} {'='*25}")
        
        try:
            wandb.init(project=config.WANDB_PROJECT_NAME, name=model_name, reinit=True)
            
            # Build dataset and tokenizer for the specific model
            dataset_builder = TextDatasetBuilder(processed_df, model_name)
            dataset, _, data_collator = dataset_builder.build()
            
            # Build model
            model = ModelBuilder(model_name).get_model()
            
            # Train model
            trainer = ModelTrainer(model, dataset, model_name, data_collator).train()
            
            # Evaluate model
            evaluator = ModelEvaluator(trainer, dataset)
            evaluator.evaluate()
        
        except Exception as e:
            print(f"ERROR during pipeline for {model_name}: {e}")
        finally:
            wandb.finish()
            print(f"{'='*25} PIPELINE FOR {model_name.upper()} COMPLETE {'='*25}\n")

if __name__ == "__main__":
    run_pipeline()
