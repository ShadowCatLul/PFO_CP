from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch

class Model:
    def __init__(self, model_name, device: str = "cpu"):
        """
        Initialize the model.
        
        Args:
            model_name (str or dict): Either a string specifying the Hugging Face model name or a path to 
                                      a local directory, or a dictionary containing paths for 'model', 'tokenizer',
                                      and 'processor' components.
            device (str): The device to load the model on (e.g., "cpu" or "cuda").
        """
        self.device = torch.device(device)
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Load components based on type of model_name
        if isinstance(model_name, dict):
            self.model_path = model_name.get("model")
            self.tokenizer_path = model_name.get("tokenizer")
            self.processor_path = model_name.get("processor")
        else:
            self.model_path = self.tokenizer_path = self.processor_path = model_name
            
        self._load_components()
        
    def _load_components(self):
        """Load model, tokenizer, and processor if available."""
        
        # Load model
        if self.model_path:
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            print("Model loaded successfully.")
        
        # Try loading tokenizer (some models may not have tokenizers)
        if self.tokenizer_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                print("Tokenizer loaded successfully.")
            except Exception as e:
                print(f"Tokenizer not found at {self.tokenizer_path}. Skipping tokenizer. Error: {e}")
        
        # Try loading processor (some models may require specific processors)
        if self.processor_path:
            try:
                self.processor = AutoProcessor.from_pretrained(self.processor_path)
                print("Processor loaded successfully.")
            except Exception as e:
                print(f"Processor not found at {self.processor_path}. Skipping processor. Error: {e}")
        
    def __call__(self, inputs):
        """Process inputs with model, handling tokenization and processing if available."""
        
        # Tokenize or process input as needed
        if self.tokenizer:
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        elif self.processor:
            inputs = self.processor(images=inputs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run model
        outputs = self.model(**inputs)
        return outputs