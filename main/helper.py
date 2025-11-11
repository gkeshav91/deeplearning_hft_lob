#!/opt/python/3.10/bin/python3
import os
import util
import pandas as pd
import numpy as np
import gc
import pickle
#--------------------------------------------------------

def load_data(directory,cols_to_replace,num_time_steps,output_file,extra_cols,ncores=8):
    os.chdir(directory)
    data_book = util.combine_csv(".", r".*.log_sample*", cores=ncores, sampling_fraction=1)
    data_fwd = util.combine_csv(".", r".*.fwdsampler*", cores=ncores, sampling_fraction=1)
    data_fwd["target_value"] = (data_fwd["mid_10.000s"] - data_fwd["mid"])*1e4/data_fwd["mid"]
    ## note : keeping the Y-value in 10^4 space helps in NN training- as else the error and gradient becomes too small.

    print(f"data_book.columns: {data_book.columns.tolist()}")
    print(f"data_fwd.columns: {data_fwd.columns.tolist()}")
    print(f"data_book dimensions: {data_book.shape}")
    print(f"data_fwd dimensions: {data_fwd.shape}")
    #--------------------------------------------------------
    #-------sequence data

    data_book["same_exchange_time_as_prev"] = (
        data_book.groupby(["date", "symbol"])["exchange_time"]
        .transform(lambda x: x == x.shift(1))
        .fillna(False)
        .astype(int)
    )

    data_book["time_gap"] = (
        data_book.groupby(["date", "symbol"])["exchange_time"]
        .diff()
        .fillna(0)
    )

    data_book["is_new"] = (data_book["tick_type"] == "NEW").astype(int)
    data_book["is_cancel"] = (data_book["tick_type"] == "CANCEL").astype(int)
    data_book["is_trade"] = (data_book["tick_type"] == "TRADE").astype(int)
    data_book["is_modify_normal"] = ((data_book["tick_type"] == "MODIFY") & (data_book["same_exchange_time_as_prev"] == 0)).astype(int)
    data_book["is_modify_hidden"] = ((data_book["tick_type"] == "MODIFY") & (data_book["same_exchange_time_as_prev"] == 1)).astype(int)  # Fixed to check for "MODIFY"
    data_book["side"] = (data_book["side"] == "BUY").astype(int)*2 - 1;

    data_book["old_quantity"] = data_book["old_quantity"].where(data_book["old_quantity"] >= 0, np.nan)
    data_book["old_price"] = data_book["old_price"].fillna(data_book["price"])
    data_book["old_quantity"] = data_book["old_quantity"].fillna(data_book["quantity"])

    #-------
    data_book[cols_to_replace] = data_book[cols_to_replace].replace([np.inf, -np.inf], np.nan)
    data_book[cols_to_replace].dropna(inplace=True)

    #-------filtering and merging data
    data_book['group_id'] = data_book.groupby(['date', 'first_tp'], sort=False).ngroup()
    group_counts = data_book['group_id'].value_counts()
    valid_group_ids = group_counts.index[group_counts.values >= num_time_steps]
    filtered = data_book[data_book['group_id'].isin(valid_group_ids)]
    del group_counts  # free memory early
    gc.collect()
    filtered = (
    #    filtered.sort_values(['group_id', 'counter'])
                filtered.groupby('group_id', group_keys=False)
                .tail(num_time_steps)
    )
    gc.collect()

    group_row_index = (
        filtered.groupby('group_id', sort=False)
                .nth(num_time_steps - 1)
                .reset_index()
    )

    join_keys = ['date', 'time', 'symbol']
    merged = pd.merge(
        group_row_index[join_keys],
        data_fwd[join_keys + ['target_value']],  # only join necessary column
        on=join_keys,
        how='left'
    )
    gc.collect()

    num_samples = len(valid_group_ids)
    feature_dim = len(cols_to_replace)
    print(f"num_samples: {num_samples}")
    print(f"feature_dim: {feature_dim}")

    X = filtered[cols_to_replace + extra_cols]
    y = merged['target_value'].to_numpy().astype(np.float32);
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    y[~np.isfinite(y)] = 0  # replaces NaN, +inf, -inf with 0
    del filtered, group_row_index, merged
    gc.collect()

    #np.savez_compressed(output_file, X=X, y=y)
    with open(output_file, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)    
    #--------------------------------------------------------


def trim_data(X,y,trimvalue):
    X = util.trimtails(X, trimvalue)
    y = np.clip(y, np.quantile(y, 1 - trimvalue), np.quantile(y, trimvalue))
    return X,y


def normalise_data(data_book,sequence_price_cols,sequence_size_cols,price_cols,size_cols):
    first_rows = data_book.groupby('group_id', sort=False).first()
    mid_price = (first_rows['ask_price0'] + first_rows['bid_price0']) / 2
    mid_size = first_rows[size_cols].sum(axis=1)
    mid_price = mid_price.replace(0, 1)
    mid_size = mid_size.replace(0, 1)

    data_book['mid_price'] = data_book['group_id'].map(mid_price)
    data_book['mid_size'] = data_book['group_id'].map(mid_size)
    data_book[price_cols] = data_book[price_cols].div(data_book['mid_price'], axis=0)
    data_book[size_cols] = data_book[size_cols].div(data_book['mid_size'], axis=0)
    data_book[sequence_price_cols] = data_book[sequence_price_cols].div(data_book['mid_price'], axis=0)
    data_book[sequence_size_cols] = data_book[sequence_size_cols].div(data_book['mid_size'], axis=0)

    data_book.drop(columns=['mid_price', 'mid_size'], inplace=True)

    # dividing by mean and std of each feature - for each (symbol,date). we can also do this for each symbol. or maybe across entire dataset. or maybe symbol, p-date
    data_book = data_book.groupby(['symbol','date']).transform( lambda x: (x - x.mean()) / x.std() )
    return data_book

