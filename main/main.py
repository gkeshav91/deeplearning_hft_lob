#!/opt/python/3.10/bin/python3
import os
import numpy as np
import numpy as np
from helper import load_data, trim_data, normalise_data
import pickle
from obalpha import print_obalpha_correlation
from model import ConvTABLNet, TABLDataset, DeepLOBRegression, LOBRegressionDataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score

#--------------------------------------------------------
# constant parameters

print("Initialising constant parameters..",flush=True)
num_levels = 10
num_time_steps = 100
trimvalue = 0.9995

obalpha_decay = 0.7
compute_obalpha = True

orderbook_cols = [
    f"{side}_{typ}{i}"
    for i in range(num_levels)
    for side, typ in [("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size")]
]

sequence_cols = [ "time_gap", "is_new", "is_cancel", "is_trade", "is_modify_normal", "is_modify_hidden", "side", "price", "quantity", "old_price", "old_quantity", "was_book_crossed", "is_cross", "is_price_added", "is_price_removed" ]
sequence_price_cols = ["price", "old_price"]
sequence_size_cols = ["quantity", "old_quantity"]
price_cols = [col for col in orderbook_cols if 'price' in col]
size_cols = [col for col in orderbook_cols if 'size' in col]
cols_to_replace = orderbook_cols + sequence_cols
extra_cols = ['group_id','symbol','date']

directory = "/workspace/keshav/projects/hftstocks/data/"
output_file = "/workspace/keshav/projects/hftstocks/data/preprocessed_data.pkl"
output_file_normalised = "/workspace/keshav/projects/hftstocks/data/preprocessed_data_normalised.npz"

#--------------------------------------------------------
# function to load and dump data to a file

def preprocess_data():
    if not os.path.exists(output_file_normalised):
        if not os.path.exists(output_file):
            load_data(directory,cols_to_replace,num_time_steps,output_file,extra_cols,ncores=16);
        with open(output_file, "rb") as f:
            data = pickle.load(f)

        data_book = data["X"]
        y = data["y"]
        print(data_book.shape)
        print(y.shape)

        #--------------------------------------------------------
        data_book = normalise_data(data_book,sequence_price_cols,sequence_size_cols,price_cols,size_cols);
        data_book = data_book.drop(columns=['group_id'])
        data_book,y = trim_data(data_book,y,trimvalue);

        num_samples = y.shape[0]
        print("num_samples: ",num_samples)
        print("num_features: ",data_book.shape[1])

        X = data_book.to_numpy().reshape(num_samples, num_time_steps, len(orderbook_cols) + len(sequence_cols)).astype(np.float32)
        X = X.transpose(0, 2, 1)  # Now shape is (N, 40+15, 100)
        X[~np.isfinite(X)] = 0  # Replace NaN, +inf, -inf with 0

        print("Final X.shape: ",X.shape)
        print("Final y.shape: ",y.shape)

        np.savez_compressed(output_file_normalised, X=X, y=y)
    else:
        print("Loading normalised data from ",output_file_normalised)
        data = np.load(output_file_normalised)
        X = data['X']
        y = data['y']
        print("Final X.shape: ",X.shape)
        print("Final y.shape: ",y.shape)

    return X,y
#--------------------------------------------------------

print("load input data..",flush=True)
X,y = preprocess_data();

#--------------------------------------------------------
# compute correlation of a standard skew expression from the LOB data

if compute_obalpha:
    print("Computing obalpha correlation for normalised data..")
    print_obalpha_correlation(X,y,num_time_steps,num_levels,orderbook_cols,obalpha_decay,is_normalised=True)
    if not os.path.exists(output_file):
        with open(output_file, "rb") as f:
            data = pickle.load(f)
        data_book = data["X"]
        y2 = data["y"]
        data_book = data_book.drop(columns=extra_cols)
        data_book,y2 = trim_data(data_book,y2,trimvalue);
        num_samples = y2.shape[0]
        X2 = data_book.to_numpy().reshape(num_samples, num_time_steps, len(orderbook_cols) + len(sequence_cols)).astype(np.float32)
        X2 = X2.transpose(0, 2, 1)  # Now shape is (N, 40+15, 100)
        X2[~np.isfinite(X2)] = 0  # Replace NaN, +inf, -inf with 0
        print("Computing obalpha correlation for raw data..")
        print_obalpha_correlation(X2,y2,num_time_steps,num_levels,orderbook_cols,obalpha_decay,is_normalised=False)

#--------------------------------------------------------
# generalised functions to train one epoch and evaluate of validation data

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model_obook, model_seq, loader, criterion, case_int,num_features1):

    if case_int == 0:
        model_obook.eval()
    elif case_int == 1:
        model_seq.eval()
    elif case_int == 3:
        model_obook.eval()
        model_seq.eval()
    elif case_int == 2:
        model_obook.eval()

    total_loss = 0
    total_absolute_error = 0.0
    input_labels = []
    output_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if case_int == 0:
                y_pred = model_obook(x[:, :num_features1, :])
            elif case_int == 1:
                y_pred = model_seq(x[:, num_features1:, :])
            elif case_int == 3:
                y_pred_obook = model_obook(x[:, :num_features1, :])
                y_pred_seq = model_seq(x[:, num_features1:, :])
                y_pred = (y_pred_obook + y_pred_seq) / (2.0)
            elif case_int == 2:
                y_pred = model_obook(x[:,:, :])

            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            total_absolute_error += torch.abs(y_pred - y).sum().item()
            input_labels.extend(y.cpu().numpy())  # Extend with NumPy array
            output_labels.extend(y_pred.cpu().numpy()) # Extend with NumPy array

    val_loss =  total_loss / len(loader.dataset)
    mae = total_absolute_error / len(loader.dataset)
    correlations = np.corrcoef(input_labels, output_labels)[0, 1]
    r2_scores = r2_score(input_labels, output_labels, multioutput='uniform_average')

    quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]  # Specify the desired quantiles
    input_quantiles = np.quantile(input_labels, quantiles)
    output_quantiles = np.quantile(output_labels, quantiles)

    print("Input Quantiles:", input_quantiles)
    print("Output Quantiles:", output_quantiles)
    return val_loss, mae, correlations, r2_scores

#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------

# here we define params and train the DeepLOB model and TABL model and dump the results on validation data.

num_features = X.shape[1]
num_features1 = 40 # hardocoded obalpha feature space.
transformed_crosssection_dim1 = num_features1 if num_features1 % 2 == 0 else num_features1 + 1
num_features2 = X.shape[1] - num_features1
transformed_crosssection_dim2 = num_features2 if num_features2 % 2 == 0 else num_features2 + 1

validation_mult = 0.2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_mult, shuffle=False)


#--------------------------------------------------------

print("running deepLOB..",flush=True)
run_deepLOB = True
if run_deepLOB:
    dropout_rate = 0.2
    batch_size = 1024
    learning_rate = 1e-3
    l2_reg = 1e-5
    num_epochs = 30

    X_train_ = X_train[:,:num_features1,:]
    X_train_ = X_train_.transpose(0, 2, 1)   # (N, 100, 55)
    print("splitting data into train and validation sets..",flush=True)
    train_dataset = LOBRegressionDataset(X_train_, y_train,num_time_steps,num_features1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    X_val_ = X_val[:,:num_features1,:]
    X_val_ = X_val_.transpose(0, 2, 1)   # (N, 100, 55)
    val_dataset = LOBRegressionDataset(X_val_, y_val,num_time_steps,num_features1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLOBRegression(dropout_rate).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!", flush=True)
        model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=l2_reg)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.5f}")
        val_loss, mae, correlations, r2_scores = evaluate(model, model, val_loader, criterion,2,num_features1)
        print(f"Val Loss: {val_loss:.5f} | MAE: {mae:.5f} | Correlation: {correlations:.5f} | R2: {r2_scores:.5f}")

    torch.save(model.state_dict(), f'/workspace/keshav/projects/hftstocks/data/model_deeplob.pth')

#--------------------------------------------------------

print("running TABL..",flush=True)
run_tabl = True
if run_tabl:
    batch_size = 16384
    dropout_rate = 0.2
    variant = 'B1'
    num_heads = 4
    transformed_time_dim = 4
    learning_rate = 1e-2
    l2_reg = 1e-2
    num_epochs = 10

    val_dataset = TABLDataset(X_val, y_val,X_val.shape[1],num_time_steps)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for i in range(3): # 3 models for 3 different feature spaces : only obalpha, only sequence, both.
        if i == 0:
            X_train_ = X_train[:,:num_features1,:]
            num_features = num_features1
            transformed_crosssection_dim = transformed_crosssection_dim1
        elif i == 1:
            X_train_ = X_train[:,num_features1:,:]
            num_features = num_features2
            transformed_crosssection_dim = transformed_crosssection_dim2
        elif i == 2:
            X_train_ = X_train
            num_features = X_train.shape[1]
            transformed_crosssection_dim = num_features if num_features % 2 == 0 else num_features + 1
        
        print("splitting data into train and validation sets..",flush=True)
        train_dataset = TABLDataset(X_train_, y_train,num_features,num_time_steps)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        #------------
        #------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ConvTABLNet(variant=variant, p_drop=dropout_rate, num_heads=num_heads, D=num_features, T=num_time_steps, Dp=transformed_crosssection_dim, Tp=transformed_time_dim).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!", flush=True)
            model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

        #------------
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=l2_reg)

        #------------
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.5f}")
            val_loss, mae, correlations, r2_scores = evaluate(model, model, val_loader, criterion,i,num_features1)
            print(f"Val Loss: {val_loss:.5f} | MAE: {mae:.5f} | Correlation: {correlations:.5f} | R2: {r2_scores:.5f}")
        
        torch.save(model.state_dict(), f'/workspace/keshav/projects/hftstocks/data/model_{i}.pth')

    #--------------------------------------------------------
    # checking combined output from seq_model and ob_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obook = ConvTABLNet(variant=variant, p_drop=dropout_rate, num_heads=num_heads, D=num_features1, T=num_time_steps, Dp=transformed_crosssection_dim1, Tp=transformed_time_dim).to(device)
    model_seq = ConvTABLNet(variant=variant, p_drop=dropout_rate, num_heads=num_heads, D=num_features2, T=num_time_steps, Dp=transformed_crosssection_dim2, Tp=transformed_time_dim).to(device)
    model_obook.load_state_dict(torch.load('/workspace/keshav/projects/hftstocks/data/model_0.pth'))
    model_seq.load_state_dict(torch.load('/workspace/keshav/projects/hftstocks/data/model_1.pth'))
    val_loss, mae, correlations, r2_scores = evaluate(model_obook, model_seq, val_loader, criterion,3,num_features1)
    print(f"Val Loss: {val_loss:.5f} | MAE: {mae:.5f} | Correlation: {correlations:.5f} | R2: {r2_scores:.5f}")
