"""
Evaluation metrics for news classification.
Computes accuracy, precision, recall, and F1-score.
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict


class ClassificationMetrics:
    """Computes classification metrics."""
    
    def __init__(self):
        pass
    
    def compute_metrics(self, y_true: List[str], y_pred: List[str], categories: List[str]) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            categories: List of all possible categories
        
        Returns:
            Dictionary with accuracy, precision, recall, and f1_score
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Compute overall accuracy
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0
        
        # Compute per-class metrics
        per_class_metrics = self._compute_per_class_metrics(y_true, y_pred, categories)
        
        # Compute macro-averaged metrics
        precisions = [m['precision'] for m in per_class_metrics.values() if m['precision'] is not None]
        recalls = [m['recall'] for m in per_class_metrics.values() if m['recall'] is not None]
        f1_scores = [m['f1_score'] for m in per_class_metrics.values() if m['f1_score'] is not None]
        
        macro_precision = np.mean(precisions) if precisions else 0.0
        macro_recall = np.mean(recalls) if recalls else 0.0
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1,
            'per_class': per_class_metrics
        }
    
    def _compute_per_class_metrics(self, y_true: List[str], y_pred: List[str], 
                                   categories: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute precision, recall, and F1 for each class.
        
        Returns:
            Dictionary mapping category to metrics dict
        """
        metrics = {}
        
        for category in categories:
            # True positives, false positives, false negatives
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == category and pred == category)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true != category and pred == category)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == category and pred != category)
            
            # Precision: TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            
            # Recall: TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            
            # F1: 2 * (precision * recall) / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = None
            
            metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': tp + fn  # Number of true instances
            }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], dataset_name: str = "", 
                     model_name: str = "", technique: str = ""):
        """Print metrics in a readable format."""
        print(f"\n{'='*70}")
        if dataset_name or model_name or technique:
            print(f"Results: {dataset_name} | {model_name} | {technique}")
            print('='*70)
        
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        if 'per_class' in metrics:
            print(f"\nPer-class metrics:")
            print(f"{'Category':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
            print('-'*70)
            
            for category, class_metrics in sorted(metrics['per_class'].items()):
                prec = f"{class_metrics['precision']:.4f}" if class_metrics['precision'] is not None else "N/A"
                rec = f"{class_metrics['recall']:.4f}" if class_metrics['recall'] is not None else "N/A"
                f1 = f"{class_metrics['f1_score']:.4f}" if class_metrics['f1_score'] is not None else "N/A"
                support = class_metrics['support']
                
                print(f"{category:<20} {prec:>10} {rec:>10} {f1:>10} {support:>10}")
        
        print('='*70)


def main():
    """Test the metrics module."""
    # Test data
    y_true = ['Sports', 'Business', 'Sports', 'Technology', 'World', 'Sports']
    y_pred = ['Sports', 'Business', 'Business', 'Technology', 'World', 'Sports']
    categories = ['Sports', 'Business', 'Technology', 'World']
    
    metrics_calc = ClassificationMetrics()
    metrics = metrics_calc.compute_metrics(y_true, y_pred, categories)
    
    metrics_calc.print_metrics(metrics, dataset_name="Test", model_name="TestModel", technique="test")


if __name__ == "__main__":
    main()
