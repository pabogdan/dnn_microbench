from .imports import *


def set_nslots():
    # Get number of cores reserved by the batch system
    # (NSLOTS is automatically set, or use 4 otherwise)
    NUMCORES = int(os.getenv("NSLOTS", 4))
    print("=" * 60)
    print("Using", NUMCORES, "core(s)")
    print("-" * 60)

    # Create TF session using correct number of cores
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
                                            intra_op_parallelism_threads=NUMCORES,
                                            allow_soft_placement=True,
                                            device_count={'CPU': NUMCORES}))

    # Set the Keras TF session
    K.set_session(sess)


def reports():
    print("=" * 60)
    print("Platform reports")
    print("-" * 60)
    print("1.{:30}:".format("Running on"), sys.platform)
    print("2.{:30}:".format("aka. os.name"), os.name)
    import psutil
    mem = psutil.virtual_memory()
    gb_norm = 1024**3
    print("3.{:30}:{:8.2f} GB".format("Total RAM", mem.total/gb_norm))
    print("4.{:30}:{:8.2f} GB".format("Available RAM", mem.available/gb_norm))


def setup_directory_structure():
    # Checking directory structure exists
    if not os.path.isdir(args.result_dir) and not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.isdir(args.model_dir) and not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)


def extract_optimizer_from_args(learning_rate=None):
    learning_rate = args.lr or learning_rate
    if args.optimizer.lower() == "sgd":
        if learning_rate:
            optimizer = keras.optimizers.SGD(lr=learning_rate)
        else:
            optimizer = keras.optimizers.SGD()
        optimizer_name = "sgd"
    elif args.optimizer.lower() in ["ada", "adadelta"]:
        optimizer = keras.optimizers.adadelta()
        optimizer_name = "adadelta"
    elif args.optimizer.lower() in ["adam"]:
        optimizer = keras.optimizers.Adam(
            lr=learning_rate,
            amsgrad=True)
        optimizer_name = "adam"
    elif args.optimizer.lower() in ["noisy_sgd", "ns"]:
        if learning_rate:
            optimizer = NoisySGD(lr=learning_rate)
        else:
            optimizer = NoisySGD()
        optimizer_name = "noisy_sgd"
    elif args.optimizer.lower() in ["rms", "rms_prop", "rmsprop"]:
        optimizer_name = "rms_prop"
        # https://github.com/Zehaos/MobileNet/blob/master/train_image_classifier.py#L307-L312
        optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    else:
        optimizer = args.optimizer
        optimizer_name = args.optimizer
    return optimizer, optimizer_name


def extract_loss_from_args():
    pass


def generate_filename(optimizer, activation,
                      sparse, loss, suffix, random_weights, acronym=False):
    _combined_string = ""
    rand_weights = "_rand" if random_weights else ""
    if not acronym:
        _combined_string += "_" + activation
        _combined_string += "_" + loss
        _combined_string += "_" + sparse
        _combined_string += "_" + optimizer + rand_weights + suffix
    else:
        _combined_string += activation[0]
        _combined_string += loss[0]
        _combined_string += "_" + sparse
        _combined_string += "_" + optimizer[0] + rand_weights + suffix
    return _combined_string


def generate_sparsity_suffix():
    if args.sparse_layers:
        if args.soft_rewiring:
            sparse_name = "sparse_soft"
        else:
            if args.conn_decay:
                sparse_name = "sparse_decay"
            else:
                sparse_name = "sparse_hard"
    else:
        sparse_name = "dense"
    return sparse_name
