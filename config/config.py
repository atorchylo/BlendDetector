class DATA:
    # Catalog and Sampling
    dataset_location = "/scratch/users/torchylo/"
    dataset_name = "NumberedGalaxyBlends_v1.0"
    catalog_path = "data/OneDegSq.fits"
    stamp_size = 25.6  # Size of the stamp, in arcseconds
    max_number = 6  # Maximum number of galaxies in a blend
    max_shift = 5.0  # Maximum shift of the galaxies from center, in arcseconds
    # Processing
    file_batch = 4  # Number of images in a single generated file
    number_cpu = 1   # Number of cpu to parallelize across
    # Dataset size
    train_batches = 25000
    valid_batches = 4000


class TRAIN:
    # datasets for the train loop
    train_data_path = '/scratch/users/torchylo/NumberedGalaxyBlends_v1.0/train'
    test_data_path = '/scratch/users/torchylo/NumberedGalaxyBlends_v1.0/valid'
    # training constants
    epochs = 50
    learning_rate = 1e-4
    batch_size = 48  # must be divisible by DATA.file_batch
    device = 'cuda'
    # network parameters
    in_ch = 6  # number of channels of the image
    concat_input = False
    num_cls = 7  # number of classes for prediction
    num_layers = 152  # depth of the neural network
    # Normalization parameters
    Q = 0.5
    S = 2731
    # logging
    log_dir = '/scratch/users/torchylo/logs/NumberDetector/'
    figures_path = '/figs'



