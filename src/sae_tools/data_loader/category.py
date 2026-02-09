"""
Category parsing and label building utilities for SAE feature analysis.

This module provides category-related functionality:
- Parse category fields (supports JSON string format)
- Build binary labels for categories
"""

import json
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np


def parse_category_field(category_str):
    """
    Parse a single category field, return the category name list
    
    Supported formats:
    - JSON string: '{"harm:XXX": 1.0, "harm:YYY": 1.0}'
    - Empty string or '{}'
    
    Args:
        category_str: category field value
        
    Returns:
        list: category name list
    """
    if not category_str or category_str == '{}':
        return []
    try:
        cat_dict = json.loads(category_str)
        return list(cat_dict.keys())
    except (json.JSONDecodeError, TypeError):
        # If not JSON format, try to process as a comma-separated string
        if isinstance(category_str, str):
            return [c.strip() for c in category_str.split(',') if c.strip()]
        return []


def build_category_labels(data_list, min_samples=50):
    """
    Build binary labels for all categories
    
    Args:
        data_list: data record list
        min_samples: minimum number of positive samples for a category
        
    Returns:
        category_labels: dict[str, np.ndarray] - {category_name: binary_labels}
        category_counts: dict[str, int] - counts for all categories
    """
    # Count all categories
    category_counts = defaultdict(int)
    for item in data_list:
        cats = parse_category_field(item.get('category', ''))
        for c in cats:
            category_counts[c] += 1
    
    # Filter categories with too few samples
    valid_categories = [c for c, cnt in category_counts.items() if cnt >= min_samples]
    
    print(f"Total {len(category_counts)} categories found")
    print(f"Among them, {len(valid_categories)} categories satisfy the minimum sample number requirement (>= {min_samples})")
    
    # Build label matrix
    category_labels = {}
    for cat in valid_categories:
        y = np.array([1 if cat in parse_category_field(item.get('category', '')) else 0 
                      for item in data_list])
        category_labels[cat] = y
    
    return category_labels, dict(category_counts)
