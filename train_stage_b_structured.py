#!/usr/bin/env python3
from disambiguation.train_stage2 import train_stage_b

if __name__ == "__main__":
    print("Training Stage B with structured cluster features + hard negative mining")
    print("=" * 80)
    
    # Train pooled ClusterMatcher with structured features
    print("\n1. Training ClusterMatcher with structured features")
    print("-" * 80)
    train_stage_b(
        window=256,
        subset="all8k",
        head="cluster",
        dropout=0.2,
        hard_neg=False,  # first without hard neg
        neg_ratio=1.0,   # 1:1 neg sampling
    )
    
    # Train with hard negative mining
    print("\n2. Training ClusterMatcher with structured features + hard negatives")
    print("-" * 80)
    train_stage_b(
        window=256,
        subset="all8k",
        head="cluster",
        dropout=0.2,
        hard_neg=True,   # enable hard neg
        neg_ratio=1.0,   # 1:1 neg sampling, but select hardest negatives
    )
