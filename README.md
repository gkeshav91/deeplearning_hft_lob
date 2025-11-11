Deep Learning models applied on HFT datasets comprising of limit order book data, and event sequence data.

The experiments are inspired from the below two research papers: 

https://arxiv.org/abs/1808.03668

https://arxiv.org/abs/1712.00975

https://arxiv.org/abs/2102.08811


Below are the results observed: 

1. Skew Signal:
Correlation between final_alpha and target_value: 0.07341

2. DeepLOB on obalpha features:
Val Loss: 26.44906 | MAE: 3.10707 | Correlation: 0.09873 | R2: 0.00925

3. Enhanced TABL: 

    a. only obalpha features : Val Loss: 26.50047 | MAE: 3.11124 | Correlation: 0.09641 | R2: 0.00733

    b. only seqalpha features : Val Loss: 26.50071 | MAE: 3.11351 | Correlation: 0.09656 | R2: 0.00732

    c. both added individually features : Val Loss: 26.47405 | MAE: 3.10693 | Correlation: 0.10995 | R2: 0.00832

    d. all features : Val Loss: 26.40126 | MAE: 3.10690 | Correlation: 0.10789 | R2: 0.01104
    

