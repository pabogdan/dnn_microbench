import ntpath
from keras_rewiring.experiments.common import *
# network generation imports
from keras_rewiring.experiments.mnist.lenet_300_100_model_setup import \
    generate_lenet_300_100_model, \
    generate_sparse_lenet_300_100_model


def train_and_test_lenet_300_100(filename):
    print(filename)
    start_time = plt.datetime.datetime.now()
    # Setting number of CPUs to use
    set_nslots()

    # Setting up directory structure
    setup_directory_structure()

    is_output_categorical = True
    dataset_info = load_and_preprocess_dataset(
        'mnist', categorical_output=is_output_categorical)
    x_train, y_train = dataset_info['train']
    x_test, y_test = dataset_info['test']
    num_classes = dataset_info['num_classes']

    # reshape input to flatten data
    x_train = x_train.reshape(x_train.shape[0], 1, np.prod(x_train.shape[1:]))
    x_test = x_test.reshape(x_test.shape[0], 1, np.prod(x_test.shape[1:]))

    print(x_train.shape)
    epochs = args.epochs or 10
    batch = 10
    learning_rate = 0.5

    # Retrieve optimizer and its name (for files and reports)
    optimizer, optimizer_name = extract_optimizer_from_args(learning_rate)

    loss = keras.losses.categorical_crossentropy

    if args.conn_level:
        builtin_sparsity = args.conn_level
    else:
        builtin_sparsity = [.01, .03, .3]
    final_conns = np.asarray(builtin_sparsity)

    conn_decay_values = None
    if args.conn_decay:
        conn_decay_values = (np.log(1. / final_conns) / epochs).tolist()
        builtin_sparsity = np.ones(len(conn_decay_values)).tolist()

    if not args.sparse_layers:
        model = generate_lenet_300_100_model(
            activation=args.activation,
            categorical_output=is_output_categorical,
            num_classes=num_classes)
    elif args.sparse_layers and not args.soft_rewiring:
        if args.conn_decay:
            print("Connectivity decay rewiring enabled", conn_decay_values)
            model = generate_sparse_lenet_300_100_model(
                activation=args.activation,
                categorical_output=is_output_categorical,
                builtin_sparsity=builtin_sparsity,
                conn_decay=conn_decay_values,
                num_classes=num_classes)
        else:
            model = generate_sparse_lenet_300_100_model(
                activation=args.activation,
                categorical_output=is_output_categorical,
                builtin_sparsity=builtin_sparsity,
                num_classes=num_classes)
    else:
        print("Soft rewiring enabled", args.soft_rewiring)
        model = generate_sparse_lenet_300_100_model(
            activation=args.activation,
            categorical_output=is_output_categorical,
            num_classes=num_classes)
    model.summary()

    # disable rewiring with sparse layers to see the performance of the layer
    # when 90% of connections are disabled and static
    deep_r = RewiringCallback(fixed_conn=args.disable_rewiring,
                              soft_limit=args.soft_rewiring,
                              noise_coeff=10 ** -5,
                              asserts_on=args.asserts_on)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

    suffix = ""
    if args.suffix:
        suffix = "_" + args.suffix

    if filename[0] == ":":
        model_name = filename[1:]
    else:
        model_name = filename

    activation_name = "relu"
    loss_name = "crossent"

    sparse_name = generate_sparsity_suffix()
    __filename = generate_filename(
        optimizer_name, activation_name, sparse_name, loss_name, suffix,
        args.random_weights)

    if args.result_filename:
        output_filename = args.result_filename
    else:
        output_filename = "results_for_" + model_name + __filename

    csv_path = os.path.join(args.result_dir, output_filename + ".csv")
    csv_logger = keras.callbacks.CSVLogger(
        csv_path,
        separator=',',
        append=False)

    callback_list = []
    if args.sparse_layers:
        callback_list.append(deep_r)

    if args.tensorboard:
        tb_log_filename = "./sparse_logs" if args.sparse_layers else "./dense_logs"

        tb = keras.callbacks.TensorBoard(
            log_dir=tb_log_filename,
            histogram_freq=1,  # turning this on needs validation_data in model.fit
            batch_size=batch, write_graph=True,
            write_grads=True, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None,
            embeddings_metadata=None, embeddings_data=None,
            update_freq='epoch',
            profile_batch='500,520')
        callback_list.append(tb)

    callback_list.append(csv_logger)
    model.fit(x_train, y_train,
              batch_size=batch,
              epochs=epochs,
              verbose=1,
              callbacks=callback_list,
              validation_data=(x_test, y_test),
              )

    score = model.evaluate(x_test, y_test, verbose=1, batch_size=batch)
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])

    end_time = plt.datetime.datetime.now()
    total_time = end_time - start_time
    print("Total time elapsed -- " + str(total_time))

    model_path = os.path.join(
        args.model_dir,
        "trained_model_of_" + model_name + __filename + ".h5")

    model.save(model_path)

    print("Results (csv) saved at", csv_path)
    print("Model saved at", model_path)
    print("Total time elapsed -- " + str(total_time))


def test_lenet_300_100(filename, no_runs):
    print(i)
    dataset_info = load_and_preprocess_dataset(
        'mnist', categorical_output=True)
    x_test, y_test = dataset_info['test']
    x_test = x_test.reshape(x_test.shape[0], 1, np.prod(x_test.shape[1:]))
    batch = 10
    # Reinstantiate the model
    c_obj = {'Sparse': Sparse,
             'SparseConv2D': SparseConv2D,
             'SparseDepthwiseConv2D': SparseDepthwiseConv2D,
             'NoisySGD': NoisySGD}
    model = keras.models.load_model(filename, custom_objects=c_obj)
    if args.test_as_dense:
        old_model_layers = model.layers
        # check if
        new_model_weights = []
        for ol in old_model_layers:
            curr_weights = ol.get_weights()
            if len(curr_weights) == 3:
                # assert that if a layer is sparse then where mask is 0 the kernel is also 0
                x = (1 - curr_weights[2]).astype(bool)
                # assert np.all(curr_weights[0][x] == 0), np.where(curr_weights[0][x] != 0)
                curr_weights[0] *= curr_weights[2]
                new_model_weights += curr_weights[:2]
            else:
                new_model_weights += curr_weights
        keras.backend.clear_session()
        model = generate_lenet_300_100_model(
            activation=args.activation,
            categorical_output=True,
            num_classes=10)
        model.set_weights(new_model_weights)
        model.compile("adam", keras.losses.categorical_crossentropy,
            metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

    print("no runs", no_runs)
    times = np.zeros(no_runs)
    scores = []
    for ni in range(no_runs):
        tb_log_filename = "./inference_logs"
        callback_list = []
        if args.tensorboard:
            tb = keras.callbacks.tensorboard_v2.TensorBoard(
                log_dir=tb_log_filename,
                batch_size=batch, write_graph=True,
                write_images=True,  histogram_freq=0,
                embeddings_freq=0, embeddings_layer_names=None,
                embeddings_metadata=None, embeddings_data=None,
                update_freq='epoch',
                profile_batch='0, 1000')
            callback_list = [tb]

        start_time = plt.datetime.datetime.now()
        score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch,
                               callbacks=callback_list)
        end_time = plt.datetime.datetime.now()
        total_time = end_time - start_time
        times[ni] = total_time.total_seconds()
        scores.append(score)
    # print('Test Loss:', score[0])
    # print('Test Accuracy:', score[1])
    # print("Total time elapsed -- " + str(total_time))
    print("mean time", np.mean(times))
    print("std time", np.std(times))
    base_name_file = str(ntpath.basename(filename))[:-3]
    csv_path = os.path.join(args.result_dir, "as_dense_times_for_" + base_name_file + ".csv")
    np.savetxt(csv_path, times, delimiter=",")

    f = plt.figure(1, figsize=(9, 9), dpi=400)
    plt.hist(times, bins=20,
             rasterized=True,
             edgecolor='k')

    plt.ylabel("Count")
    plt.xlabel("Inference duration (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir,
                             "hist_times_for_" + base_name_file + ".png"))
    plt.close(f)
    return scores, times


if __name__ == "__main__":
    for i in args.model:
        if args.just_test:
            no_runs = 10
            s, t = test_lenet_300_100(i, no_runs)
            print("The score are:", s)
        else:
            train_and_test_lenet_300_100(i)
