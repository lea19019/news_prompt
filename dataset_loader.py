"""
Dataset loader and preprocessor for news classification experiments.
Downloads, preprocesses, and caches datasets in a consistent format.
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
import requests
from io import StringIO


class DatasetLoader:
    """Handles downloading and preprocessing of news classification datasets."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            'ag_news': {
                'categories': ['World', 'Sports', 'Business', 'Sci/Tech'],
                'train_file': 'ag_news_train.csv',
                'test_file': 'ag_news_test.csv'
            },
            'bbc_news': {
                'categories': ['business', 'entertainment', 'politics', 'sport', 'tech'],
                'train_file': 'bbc_news_train.csv',
                'test_file': 'bbc_news_test.csv'
            },
            '20newsgroups': {
                'categories': None,  # Will be set dynamically
                'train_file': '20newsgroups_train.csv',
                'test_file': '20newsgroups_test.csv'
            }
        }
    
    def load_dataset(self, dataset_name, split='test'):
        """
        Load a preprocessed dataset.
        
        Args:
            dataset_name: One of 'ag_news', 'bbc_news', '20newsgroups'
            split: 'train' or 'test'
        
        Returns:
            pd.DataFrame with columns: text, label, label_text
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        file_key = f'{split}_file'
        file_path = self.data_dir / config[file_key]
        
        # Download and preprocess if not cached
        if not file_path.exists():
            print(f"Dataset not found. Downloading {dataset_name}...")
            self._download_and_preprocess(dataset_name)
        
        df = pd.read_csv(file_path)
        return df
    
    def get_categories(self, dataset_name):
        """Get the list of categories for a dataset."""
        df = self.load_dataset(dataset_name, split='test')
        return sorted(df['label_text'].unique().tolist())
    
    def get_few_shot_examples(self, dataset_name, n=3, split='train'):
        """
        Get few-shot examples for a dataset.
        Selects one example per category (up to n examples total).
        
        Args:
            dataset_name: Name of the dataset
            n: Number of examples to return
            split: Which split to sample from ('train' or 'test')
        
        Returns:
            List of (text, label_text) tuples
        """
        df = self.load_dataset(dataset_name, split=split)
        categories = self.get_categories(dataset_name)
        
        examples = []
        for category in categories[:n]:
            # Get one example from this category
            cat_examples = df[df['label_text'] == category]
            if len(cat_examples) > 0:
                sample = cat_examples.sample(1, random_state=42).iloc[0]
                # Truncate text for brevity in prompts
                text = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
                examples.append((text, sample['label_text']))
        
        # If we need more examples, sample randomly
        while len(examples) < n:
            sample = df.sample(1, random_state=42 + len(examples)).iloc[0]
            text = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
            examples.append((text, sample['label_text']))
        
        return examples[:n]
    
    def _download_and_preprocess(self, dataset_name):
        """Download and preprocess a specific dataset."""
        if dataset_name == 'ag_news':
            self._process_ag_news()
        elif dataset_name == 'bbc_news':
            self._process_bbc_news()
        elif dataset_name == '20newsgroups':
            self._process_20newsgroups()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _process_ag_news(self):
        """Download and process AG News dataset."""
        print("Downloading AG News dataset...")
        
        # AG News is available through Hugging Face datasets or direct CSV
        # Using direct CSV URLs from the original source
        base_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
        
        for split in ['train', 'test']:
            url = base_url + f"{split}.csv"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # AG News format: "class","title","description"
                df = pd.read_csv(StringIO(response.text), names=['label', 'title', 'description'])
                
                # Combine title and description
                df['text'] = df['title'] + ". " + df['description']
                
                # Map labels to category names (1-indexed in AG News)
                label_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
                df['label_text'] = df['label'].map(label_map)
                df['label'] = df['label'] - 1  # Convert to 0-indexed
                
                # Keep only necessary columns
                df = df[['text', 'label', 'label_text']]
                
                # Save
                output_file = self.data_dir / self.dataset_configs['ag_news'][f'{split}_file']
                df.to_csv(output_file, index=False)
                print(f"Saved {split} split: {len(df)} samples")
                
            except Exception as e:
                print(f"Error downloading AG News {split} split: {e}")
                print("You may need to download it manually.")
                raise
    
    def _process_bbc_news(self):
        """Download and process BBC News dataset."""
        print("Downloading BBC News dataset...")
        
        # Try multiple sources for BBC News dataset
        urls_to_try = [
            # GitHub mirror 1
            "https://raw.githubusercontent.com/zackthoutt/bbc-news-classification/master/bbc-text.csv",
            # GitHub mirror 2  
            "https://raw.githubusercontent.com/Pratik94229/BBC-News-Classification/master/bbc-text.csv",
            # Alternative direct source
            "https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv",
        ]
        
        df = None
        for url in urls_to_try:
            try:
                print(f"Trying: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                df = pd.read_csv(StringIO(response.text))
                print(f"✓ Successfully downloaded from {url}")
                break
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        if df is None:
            print("\n⚠️  Could not download BBC News dataset from any source.")
            print("Please download manually from: https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive")
            print("\nManual setup instructions:")
            print("1. Download 'bbc-text.csv' from Kaggle")
            print("2. Place it in the data/ folder")
            print("3. Run this script again")
            raise Exception("BBC News dataset unavailable from automatic sources")
        
        try:
            # Standardize column names
            # Common formats: 'text'/'category' or 'Text'/'Category'
            if 'category' in df.columns and 'text' in df.columns:
                df = df.rename(columns={'category': 'label_text'})
            elif 'Category' in df.columns and 'Text' in df.columns:
                df = df.rename(columns={'Text': 'text', 'Category': 'label_text'})
            elif 'Category' in df.columns and 'text' in df.columns:
                df = df.rename(columns={'Category': 'label_text'})
            
            # Create numeric labels
            categories = sorted(df['label_text'].unique())
            label_map = {cat: idx for idx, cat in enumerate(categories)}
            df['label'] = df['label_text'].map(label_map)
            
            # Keep only necessary columns
            df = df[['text', 'label', 'label_text']]
            
            # Split into train/test (80/20)
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)
            
            # Save
            train_file = self.data_dir / self.dataset_configs['bbc_news']['train_file']
            test_file = self.data_dir / self.dataset_configs['bbc_news']['test_file']
            
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            
            print(f"Saved train split: {len(train_df)} samples")
            print(f"Saved test split: {len(test_df)} samples")
            
        except Exception as e:
            print(f"Error processing BBC News: {e}")
            print("\n⚠️  Please download manually from: https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive")
            raise
    
    def _process_20newsgroups(self):
        """Download and process 20 Newsgroups dataset."""
        print("Downloading 20 Newsgroups dataset...")
        
        try:
            # Use sklearn's built-in loader
            for split in ['train', 'test']:
                subset = 'train' if split == 'train' else 'test'
                newsgroups = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
                
                df = pd.DataFrame({
                    'text': newsgroups.data,
                    'label': newsgroups.target,
                    'label_text': [newsgroups.target_names[i] for i in newsgroups.target]
                })
                
                # Save
                output_file = self.data_dir / self.dataset_configs['20newsgroups'][f'{split}_file']
                df.to_csv(output_file, index=False)
                print(f"Saved {split} split: {len(df)} samples")
                
            # Update categories
            self.dataset_configs['20newsgroups']['categories'] = newsgroups.target_names
            
        except Exception as e:
            print(f"Error downloading 20 Newsgroups: {e}")
            raise


def main():
    """Download and preprocess all datasets."""
    loader = DatasetLoader()
    
    datasets = ['ag_news', 'bbc_news', '20newsgroups']
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print('='*60)
        
        try:
            # This will download if not cached
            train_df = loader.load_dataset(dataset_name, split='train')
            test_df = loader.load_dataset(dataset_name, split='test')
            
            print(f"Train samples: {len(train_df)}")
            print(f"Test samples: {len(test_df)}")
            print(f"Categories: {loader.get_categories(dataset_name)}")
            print(f"Example few-shot samples:")
            for text, label in loader.get_few_shot_examples(dataset_name, n=3):
                print(f"  - '{text[:80]}...' → {label}")
            
        except Exception as e:
            print(f"Failed to process {dataset_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print('='*60)


if __name__ == "__main__":
    main()
