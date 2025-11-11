#!/opt/python/3.10/bin/python3
import pandas as pd
import numpy as np

#--------------------------------------------------------
def print_obalpha_correlation(X: np.ndarray, y: np.ndarray,num_time_steps: int,num_levels: int, orderbook_cols, alpha_decay, is_normalised=True) -> pd.DataFrame:

    # Map column name -> index
    col2idx = {c:i for i,c in enumerate(orderbook_cols)}

    # ------ Weighted mid using exponentially decaying level weights ------
    lvl_weights = np.array([2.0**(-i) for i in range(num_levels)], dtype=np.float64)
    lvl_weights = lvl_weights / lvl_weights.sum()  # (L,)

    # Extract bid/ask price ladders: shapes (N, L, T)
    bid_prices = np.stack([X[:, col2idx[f"bid_price{i}"], :] for i in range(num_levels)], axis=1)
    ask_prices = np.stack([X[:, col2idx[f"ask_price{i}"], :] for i in range(num_levels)], axis=1)

    # Level-weighted bid/ask: shapes (N, T)
    weighted_bid = np.einsum('l,nlt->nt', lvl_weights, bid_prices)
    weighted_ask = np.einsum('l,nlt->nt', lvl_weights, ask_prices)

    mid_pred = 0.5 * (weighted_bid + weighted_ask)              # (N, T)
    mid_price0 = 0.5 * (X[:, col2idx["bid_price0"], :] + X[:, col2idx["ask_price0"], :])  # (N, T)

    # Alpha time-series per sample
    if is_normalised :
        alpha_ts = (mid_pred - mid_price0);                      # (N, T)
    else :
        eps = 1e-8  # small constant to avoid division by zero
        denom = np.where(np.abs(mid_price0) < eps, np.nan, mid_price0)
        alpha_ts = (mid_pred - mid_price0) / denom
        alpha_ts = np.nan_to_num(alpha_ts, nan=0.0, posinf=0.0, neginf=0.0)

    # ------ Exponentially weighted average over the last 100 steps ------
    # Newer timestamps get higher weight (rightmost = most recent)
    t_weights = np.array([alpha_decay ** i for i in range(num_time_steps-1, -1, -1)], dtype=np.float64)
    t_weights = t_weights / t_weights.sum()                      # (T,)

    final_alpha = np.einsum('nt,t->n', alpha_ts, t_weights)      # (N,)
    alpha_last = alpha_ts[:, -1]                                 # (N,)

    # Build merged-like output
    merged = pd.DataFrame({
        "final_alpha": final_alpha,
        "alpha": alpha_last,
        "target_value": y.astype(np.float64, copy=False),
    })

    # Clean NaN/inf -> 0
    for col in ["final_alpha", "alpha", "target_value"]:
        merged[col] = np.nan_to_num(merged[col], nan=0.0, posinf=0.0, neginf=0.0)

    correlations = np.corrcoef(merged['final_alpha'], merged['target_value'])[0, 1]
    print(f"Correlation between final_alpha and target_value: {correlations:.5f}")
    correlations = np.corrcoef(merged['alpha'], merged['target_value'])[0, 1]
    print(f"Correlation between alpha and target_value: {correlations:.5f}")

    return
