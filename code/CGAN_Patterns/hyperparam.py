import pandas as pd
import numpy as np

class Hyperparam:

    trial_folder_prefix='P_RESP_'
    TS_UNIFIED = 180  # Unified Sampling for all datasets in seconds
    MODE = 'vanilla'  # 'wgan','vanilla', 'ian_vanilla',logit,'wgan-gp-smk', lip-mul', 'lip-log', 'lip-sum'
    APPLIED_CLEANING = 'filtered'  # Either matched-filtered (filtered) or matched-filtered then Kolmogorov–Smirnov test
    ADNTL_FEATURES = 3  # If 0, no featre are added. If this is =1, then histogram counts for highest power is considered.
    # If it is three , counts for all three bins will be considered (Note histogram is calculated for each training example)
    HISTOGRAM_BINS = 3  # Bins of histograms  that is calculated for each training example
    RPTD_EXAMPLES =1   # Repeated rows. 1 means NO repeated examples are generated
    SHFT_STEPS = 0  # stride to genertae repeated examples. 0 means no shift is applied
    MB_SIZE = 512  # Mini Batch Size (i.e. batch)
    NOISE='normal' #noise either "normal": normal N(0,1) or "uniform": uniform (-1,1)
    Z_DIM = 100  # Noise Dimension #Incresing nois dim may lead to early divergence
    D_H_DIM = [100, 150]  # Disc number of nodes in hidden layers.
    G_H_DIM = [100, 150]  # Generator number of nodes in hidden layers.
    EPOCHS = 2001
    E_H_DIM = [8, 10, 10, 10]  # Nodes in Evaluator
    LAMBDA = 0.01  # Gradient penalty lambda hyperparameter
    LAMBDA_G = 0  # Gradient penalty lambda hyperparameter
    WEIGHTS_INIT = 'xavier'  # xavier or normal
    STDDEV = 0.05
    D_KEEP_PERC = 0.98  # dropout for Disc (1= no drop) #Keep 0.95 or below to prevent overfitting of Generator (i.e. generating
    # exactly the same output)
    G_KEEP_PERC = 1
    D_STEPS = 3  # No. of steps to apply to discriminator #high value causes inf (i.e. iterations do not converge)
    G_STEPS = 3  # No. of steps to apply to generator #high value may causes mode collapse (synthetic images are similar to eah other)

