class Hyperparam:

    trial_folder_prefix='H_'
    PICKED_FEATURE='all'
    FEATURES=['week','day','hour']
    FEATURES_NAMES=['Week of Year','Day of Week','Hour of Day']
    FEATURES_COLORS=['orange', 'magenta', 'cyan']
    FEATURES_BINS=[52,7,24]
    FEATURES_RANGES=[[0,51],[0,6],[0,23]]
    APPLIED_CLEANING = 'filtered'
    MODE = 'vanilla'
    if PICKED_FEATURE == 'all':
        features = 3
    else:
        features = 1
    SAMPLES = 1000
    MB_SIZE = 512
    NOISE = 'uniform'
    Z_DIM = 50
    D_H_DIM = [100, 200,2]
    G_H_DIM = [120, 240,120]
    EPOCHS = 5001
    LAMBDA = 0.01
    LAMBDA_G = 0.02
    WEIGHTS_INIT = 'xavier'
    STDDEV = 0.05
    D_KEEP_PERC = 0.95
    G_KEEP_PERC = 1
    D_STEPS = 7
    G_STEPS = 10
    SIGMAS = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]



