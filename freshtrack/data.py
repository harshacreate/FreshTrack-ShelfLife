"""Synthetic regression data for FreshTrack-ShelfLife — Food Shelf-Life Estimation Prototype."""
from __future__ import annotations
import numpy as np
FEATURES=['product_code', 'mean_temp', 'temp_excursion_hours', 'humidity', 'initial_moisture', 'packaging', 'handling_days']
def make_dataset(n=2400,seed=42):
    if n<160:raise ValueError("n must be at least 160")
    rng=np.random.default_rng(seed)
    batches=max(30,n//30);batch=np.arange(n)%batches;product=rng.integers(0,5,n).astype(float);opt=np.array([4,8,12,18,22]);mean_temp=opt[product.astype(int)]+rng.normal(0,5,n);excursion=np.maximum(0,np.abs(mean_temp-opt[product.astype(int)])-3)*rng.uniform(1,5,n);humidity=np.clip(rng.normal(65,15,n),20,100);moisture=np.clip(rng.normal(.35,.12,n),.05,.8);packaging=rng.integers(0,3,n).astype(float);handling=np.clip(rng.gamma(1.5,2,n),0,12);base=np.array([14,10,8,6,5])[product.astype(int)];protection=np.array([0,1.5,3])[packaging.astype(int)];y=np.maximum(.2,base-protection-.24*excursion-.025*np.abs(humidity-60)-4*moisture-.35*handling+rng.normal(0,1.8,n));groups=batch
    X=np.column_stack([product, mean_temp, excursion, humidity, moisture, packaging, handling]).astype(float)
    if X.shape!=(n,len(FEATURES)) or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError("invalid generated regression data")
    return X,np.asarray(y,dtype=float),np.asarray(groups)
