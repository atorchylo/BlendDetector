class DATA:
    # Catalog and Sampling
    catalog_path = "data/OneDegSq.fits"
    stamp_size = 25.6  # Size of the stamp, in arcseconds
    max_number = 6  # Maximum number of galaxies in a blend
    max_shift = 5.0  # Maximum shift of the galaxies from center, in arcseconds
    # Processing
    file_batch = 4  # Number of images in a single generated file
    number_cpu = 1   # Number of cpu to parallelize across
    # Dataset size
    train_batches = 50000
    valid_batches = 500


class TRAIN:
    # datasets for the train loop
    train_data_path = 'data/NumberedGalaxyBlends/train'
    test_data_path = 'data/NumberedGalaxyBlends/valid'
    # training constants
    epochs = 2
    learning_rate = 1e-5
    batch_size = 16  # must be divisible by DATA.file_batch
    device = 'cpu'
    # network parameters
    in_ch = 6  # number of channels on the input
    num_cls = 6  # number of classes for prediction
    # Normalization parameters
    Q = 0.5
    S = 2731
    # logging
    log_dir = 'logs/NumberDetector/'



