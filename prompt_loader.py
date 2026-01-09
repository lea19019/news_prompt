"""
Prompt template loader and formatter.
Loads prompt templates from the prompts/ directory and formats them with dataset-specific parameters.
"""

from pathlib import Path
from typing import List, Tuple

from enriched_categories import get_enriched_categories_text


class PromptLoader:
    """Handles loading and formatting of prompt templates."""
    
    def __init__(self, prompts_dir="prompts"):
        self.prompts_dir = Path(prompts_dir)
        
        # Map technique names to template files
        self.prompt_templates = {
            'zero_shot': 'zero_shot.txt',
            'few_shot': 'few_shot.txt',
            'few_shot_3': 'few_shot_3.txt',
            'few_shot_5': 'few_shot_5.txt',
            'constrained': 'constrained.txt',
            'chain_of_thought': 'chain_of_thought.txt',
            'self_consistency': 'self_consistency.txt',
            'zero_shot_enriched': 'zero_shot_enriched.txt',
            'few_shot_3_enriched': 'few_shot_3_enriched.txt',
            'few_shot_5_enriched': 'few_shot_5_enriched.txt',
            'chain_of_thought_enriched': 'chain_of_thought_enriched.txt'
        }
    
    def load_template(self, technique: str) -> str:
        """
        Load a prompt template from file.
        
        Args:
            technique: One of the prompting techniques
        
        Returns:
            Template string with placeholders
        """
        if technique not in self.prompt_templates:
            raise ValueError(f"Unknown technique: {technique}")
        
        template_file = self.prompts_dir / self.prompt_templates[technique]
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")
        
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def format_prompt(self, 
                     technique: str,
                     text: str,
                     categories: List[str],
                     examples: List[Tuple[str, str]] = None,
                     dataset_name: str = None) -> str:
        """
        Format a prompt template with specific parameters.
        
        Args:
            technique: Prompting technique to use
            text: The news article text to classify
            categories: List of valid category names
            examples: List of (text, label) tuples for few-shot examples
            dataset_name: Dataset name for enriched categories (optional)
        
        Returns:
            Formatted prompt string
        """
        template = self.load_template(technique)
        
        # Check if this is an enriched technique
        is_enriched = '_enriched' in technique
        
        # Format categories
        if is_enriched and dataset_name:
            categories_str = get_enriched_categories_text(dataset_name, categories)
        else:
            categories_str = ', '.join(categories)
        
        # Format examples if provided
        examples_str = ""
        if examples:
            examples_str = self._format_examples(examples)
        
        # Replace placeholders
        prompt = template.replace('{text}', text)
        prompt = prompt.replace('{categories}', categories_str)
        prompt = prompt.replace('{enriched_categories}', categories_str)
        
        if '{examples}' in prompt:
            if not examples:
                # If template expects examples but none provided, use empty string
                prompt = prompt.replace('{examples}', '')
            else:
                prompt = prompt.replace('{examples}', examples_str)
        
        return prompt
    
    def _format_examples(self, examples: List[Tuple[str, str]]) -> str:
        """
        Format few-shot examples in a consistent style.
        
        Args:
            examples: List of (text, label) tuples
        
        Returns:
            Formatted examples string
        """
        formatted = []
        for text, label in examples:
            formatted.append(f'- "{text}" → {label}')
        
        return '\n'.join(formatted)
    
    def get_available_techniques(self) -> List[str]:
        """Return list of available prompting techniques."""
        return list(self.prompt_templates.keys())


def main():
    """Test the prompt loader."""
    loader = PromptLoader()
    
    # Test data
    test_text = "Lakers win NBA championship against Miami Heat"
    test_categories = ['Sports', 'Business', 'Technology', 'World']
    test_examples = [
        ("Federal Reserve raises interest rates", "Business"),
        ("New AI model breaks records", "Technology"),
        ("UN climate summit concludes", "World")
    ]
    
    print("Available techniques:", loader.get_available_techniques())
    print("\n" + "="*60)
    
    for technique in loader.get_available_techniques():
        print(f"\n{technique.upper()}:")
        print("-" * 60)
        
        try:
            if technique == 'zero_shot':
                prompt = loader.format_prompt(technique, test_text, test_categories)
            else:
                prompt = loader.format_prompt(technique, test_text, test_categories, test_examples)
            
            print(prompt)
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 60)


if __name__ == "__main__":
    main()
