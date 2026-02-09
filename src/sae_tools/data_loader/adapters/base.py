import os
import json
import logging
from typing import Optional, Any, List, Dict, Union

logger = logging.getLogger("BenchmarkRunner")


def make_category(
    harm_labels: List[Union[str, tuple]] = None, 
    metadata: Dict[str, Any] = None
) -> str:
    """build a unified format of category JSON string
    
    Args:
        harm_labels: harm categories list, supports two formats:
            - string list: ["Violence", "Sexual"] → default value 1.0
            - tuple list: [("Violence", 0.8), ("Sexual", 0.6)] → custom confidence
        metadata: metadata dictionary, supports multiple value types:
            - bool: {"adversarial": True} → 1.0
            - float: {"confidence": 0.85} → directly use
            - str: {"focus": "kill"} → as key suffix, value is 1.0
    
    Returns:
        unified format of category JSON string, all values are float [0,1]
        example: '{"harm:Violence": 1.0, "meta:adversarial": 1.0}'
        
    Note:
        use JSON string instead of dict, to avoid the problem of missing values being set to None
        by datasets.map() automatically aligning different sample dictionaries.
    """
    result = {}
    
    if harm_labels:
        for item in harm_labels:
            if item:
                if isinstance(item, tuple):
                    # tuple format: (label, confidence)
                    label, confidence = item
                    if label:
                        result[f"harm:{label}"] = float(max(0.0, min(1.0, confidence)))
                else:
                    # string format: default confidence 1.0
                    result[f"harm:{item}"] = 1.0
    
    if metadata:
        for key, value in metadata.items():
            if value is None or value == "":
                continue
            if isinstance(value, bool):
                if value:  # True → 1.0, False → skip
                    result[f"meta:{key}"] = 1.0
            elif isinstance(value, (int, float)):
                # numeric type: normalize to [0,1]
                result[f"meta:{key}"] = float(max(0.0, min(1.0, value)))
            else:
                # string etc. other types: as key suffix, value is 1.0
                result[f"meta:{key}:{value}"] = 1.0
    
    return json.dumps(result, ensure_ascii=False)


def parse_category(category_str: str) -> Dict[str, float]:
    """parse category JSON string to dictionary
    
    Args:
        category_str: category JSON string
    
    Returns:
        category dictionary, keys are "harm:xxx" or "meta:xxx", values are float [0,1]
    """
    if not category_str:
        return {}
    try:
        return json.loads(category_str)
    except (json.JSONDecodeError, TypeError):
        return {}


class BaseAdapter:
    # Default configuration
    SPLIT: Optional[str] = None
    SUBSET: Optional[str] = None

    def load(self, dataset_path: str, max_samples: Optional[int] = None, 
             subset: Optional[str] = None, split: Optional[str] = None) -> Any:
        
        actual_subset = subset if subset is not None else self.SUBSET
        actual_split = split if split is not None else self.SPLIT
        
        logger.info(f"📚 Loading dataset: {self.__class__.__name__}")
        logger.info(f"   Path: {dataset_path}")
        logger.info(f"   Split: {actual_split}, Subset: {actual_subset}")

        try:
            from datasets import (
                load_dataset,
                Dataset,
                DatasetDict,
            )

            # directly use HF library to load, let it handle the errors
            ds = load_dataset(path=dataset_path, split=actual_split, name=actual_subset)

            # Slicing
            if max_samples and max_samples > 0:
                if isinstance(ds, Dataset):
                    total_rows = ds.num_rows
                    logger.info(
                        f"✂️  Slicing dataset to first {max_samples}  / {total_rows} samples."
                    )
                    limit = min(total_rows, max_samples)
                    ds = ds.select(range(limit))
                else:
                    logger.info(
                        "Dataset is not a regular Dataset; skipping slicing for debug."
                    )

            # Apply Transform
            if isinstance(ds, DatasetDict):
                original_cols = sorted(
                    {c for cols in ds.column_names.values() for c in cols}
                )
            elif isinstance(ds, Dataset):
                original_cols = ds.column_names
            else:
                original_cols = None

            logger.info("🔄 Applying transform...")

            ds_adapted = ds.map(
                self.transform,
                remove_columns=original_cols,
                load_from_cache_file=False,  # disable cache, ensure using the latest transform logic
            )
            return ds_adapted

        except Exception as e:
            logger.error(f"❌ Error loading {dataset_path}: {e}")
            raise e

    def peek(self, dataset_path: str, subset: Optional[str] = None, split: Optional[str] = None) -> None:
        """load the first sample, show all original fields information"""
        from datasets import load_dataset

        # use the parameters passed in, if None, use the class attribute default value
        actual_subset = subset if subset is not None else self.SUBSET
        actual_split = split if split is not None else self.SPLIT

        # load the dataset (only the first sample)
        ds = load_dataset(path=dataset_path, split=actual_split, name=actual_subset)
        example = ds[0]

        # pretty print
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_path}")
        print(f"Split: {actual_split}, Subset: {actual_subset}")
        print(f"{'='*60}\n")

        for key, value in example.items():
            value_type = type(value).__name__
            # truncate long text (more than 100 characters)
            display_value = str(value)
            if len(display_value) > 100:
                display_value = display_value[:100] + "..."
            print(f"[{key}] ({value_type})")
            print(f"  {display_value}\n")

    def transform(self, example):
        raise NotImplementedError
