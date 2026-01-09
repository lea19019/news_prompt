"""
Model loader and inference wrapper for Ollama.
Handles model loading and prediction generation with different configurations.
"""

import ollama
import re
from typing import List, Dict, Any
from collections import Counter


class ModelInference:
    """Wrapper for Ollama model inference."""
    
    def __init__(self):
        self.available_models = None
        self._refresh_models()
    
    def _refresh_models(self):
        """Refresh the list of available models."""
        try:
            models_response = ollama.list()
            # Handle ListResponse object with models attribute
            if hasattr(models_response, 'models'):
                self.available_models = [m.model for m in models_response.models]
            # Handle dict with 'models' key
            elif isinstance(models_response, dict) and 'models' in models_response:
                self.available_models = [m.get('name', m.get('model', '')) for m in models_response['models']]
            # Handle direct list
            elif isinstance(models_response, list):
                self.available_models = [m.get('name', m.get('model', '')) for m in models_response]
            else:
                self.available_models = []
        except Exception as e:
            print(f"Warning: Could not fetch models list: {e}")
            self.available_models = []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally."""
        return model_name in self.available_models
    
    def get_available_models(self) -> List[str]:
        """Return list of available models."""
        return self.available_models.copy()
    
    def generate(self, 
                model_name: str,
                prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 100) -> str:
        """
        Generate a response from the model.
        
        Args:
            model_name: Name of the Ollama model
            prompt: The prompt to send to the model
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text response
        """
        try:
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Error generating response from {model_name}: {e}")
            return ""
    
    def predict_with_self_consistency(self,
                                     model_name: str,
                                     prompt: str,
                                     n_samples: int = 5,
                                     temperature: float = 0.7,
                                     categories: List[str] = None) -> str:
        """
        Use self-consistency: generate multiple predictions and take majority vote.
        
        Args:
            model_name: Name of the Ollama model
            prompt: The prompt to send to the model
            n_samples: Number of samples to generate
            temperature: Sampling temperature for diversity
            categories: Valid categories for parsing predictions
        
        Returns:
            Majority-voted category
        """
        predictions = []
        
        for i in range(n_samples):
            response = self.generate(model_name, prompt, temperature=temperature)
            parsed = self.parse_category(response, categories)
            if parsed:
                predictions.append(parsed)
        
        if not predictions:
            return "unknown"
        
        # Majority vote
        vote_counts = Counter(predictions)
        majority_category, _ = vote_counts.most_common(1)[0]
        
        return majority_category
    
    def parse_category(self, response: str, categories: List[str]) -> str:
        """
        Parse model response to extract predicted category.
        
        Args:
            response: Raw model response
            categories: List of valid category names
        
        Returns:
            Extracted category name or None if parsing fails
        """
        if not response:
            return None
        
        # Clean response
        response = response.strip()
        
        # Strategy 1: Direct match (case-insensitive)
        response_lower = response.lower()
        for category in categories:
            if category.lower() == response_lower:
                return category
        
        # Strategy 2: Check if category appears in response
        for category in categories:
            if category.lower() in response_lower:
                return category
        
        # Strategy 3: Extract first word/line
        first_line = response.split('\n')[0].strip()
        first_word = first_line.split()[0].strip('.,!?:"\'').lower()
        
        for category in categories:
            if category.lower() == first_word or category.lower() == first_line.lower():
                return category
        
        # Strategy 4: Fuzzy matching for common variations
        # Handle variations like "sci/tech" vs "sci-tech" vs "science/technology"
        category_variations = {
            'sci/tech': ['scien', 'tech', 'sci'],
            'science/technology': ['scien', 'tech', 'sci'],
            'sci-tech': ['scien', 'tech', 'sci'],
            'business': ['busi', 'econ', 'financ'],
            'entertainment': ['entertain', 'ent'],
            'politics': ['politic', 'polit']
        }
        
        for category in categories:
            cat_lower = category.lower()
            if cat_lower in category_variations:
                for variation in category_variations[cat_lower]:
                    if variation in response_lower:
                        return category
        
        # If all strategies fail, return None
        return None
    
    def predict(self,
               model_name: str,
               prompt: str,
               categories: List[str],
               technique: str = 'standard',
               temperature: float = 0.0) -> Dict[str, Any]:
        """
        Generate a prediction with the specified technique.
        
        Args:
            model_name: Name of the Ollama model
            prompt: The formatted prompt
            categories: Valid category names
            technique: Prompting technique (affects temperature and sampling)
            temperature: Override temperature
        
        Returns:
            Dictionary with 'prediction', 'raw_response', and metadata
        """
        # Use self-consistency for that technique
        if technique == 'self_consistency':
            prediction = self.predict_with_self_consistency(
                model_name, prompt, n_samples=5, temperature=0.7, categories=categories
            )
            return {
                'prediction': prediction,
                'raw_response': 'self_consistency_ensemble',
                'technique': technique,
                'temperature': 0.7,
                'n_samples': 5
            }
        else:
            # Standard single prediction
            raw_response = self.generate(model_name, prompt, temperature=temperature)
            prediction = self.parse_category(raw_response, categories)
            
            return {
                'prediction': prediction if prediction else 'unknown',
                'raw_response': raw_response,
                'technique': technique,
                'temperature': temperature,
                'n_samples': 1
            }


def main():
    """Test the model inference."""
    inference = ModelInference()
    
    print("Available models:")
    for model in inference.get_available_models():
        print(f"  - {model}")
    
    if not inference.available_models:
        print("\nNo models available. Please install models using Ollama.")
        print("Example: ollama pull gemma3:4b")
        return
    
    # Test with first available model
    test_model = inference.available_models[0]
    test_prompt = "Classify this article: Lakers win NBA championship. Category: Sports, Business, Technology, World"
    test_categories = ['Sports', 'Business', 'Technology', 'World']
    
    print(f"\nTesting with model: {test_model}")
    print(f"Prompt: {test_prompt}")
    
    result = inference.predict(test_model, test_prompt, test_categories)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Raw response: {result['raw_response']}")


if __name__ == "__main__":
    main()
