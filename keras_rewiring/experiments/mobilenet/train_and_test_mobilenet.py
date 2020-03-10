from keras_rewiring.experiments.common import *


def train_and_test_mobilenet():
    start_time = plt.datetime.datetime.now()
    # Setting number of CPUs to use
    set_nslots()

    # Setting up directory structure
    setup_directory_structure()

    # Print some reports
    reports()

    epochs = args.epochs or 10
    if args.continue_from_epoch:
        epochs += args.continue_from_epoch
    # https://github.com/Zehaos/MobileNet/blob/master/train_image_classifier.py#L191-L192
    # batch = 32
    # Never mind. According to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1_train.py
    # batch = 64
    batch = args.batch or 64
    learning_rate = args.lr or 1e-4  # the default is None from argparser

    p_0 = args.conn_level or .10  # global connectivity level
    print("Flat connectivity level", p_0)
    builtin_sparsity = [p_0] * 8
    alphas = [0, 10 ** -7, 10 ** -6, 10 ** -9, 0]
    final_conns = np.asarray(builtin_sparsity)
    conn_decay_values = None
    if args.conn_decay:
        conn_decay_values = (np.log(1. / final_conns) / epochs).tolist()
        builtin_sparsity = np.ones(len(conn_decay_values)).tolist()

    # Check whether the model that has been provided to argparser is .hdf5 / .h5 on
    # disk or a reference to Keras Mobilenet (i.e. :mobilenet)
    _is_builtin_model = False

    # Add LR reduction schedule based on Inception paper

    def lr_reduction_schedule(epoch, lr):
        """
        a function that takes an epoch index as input (integer, indexed from 0)
        and current learning rate and
        returns a new learning rate as output (float).
        :param epoch: epoch index (indexed from 0)
        :type epoch: int
        :param lr: current learning rate
        :type lr: float
        :return: new learning rate
        :rtype: float
        """
        if epoch % 7 == 0:
            return lr * .96
        return lr

    if args.continue_from_epoch != 0:
        for _previous_epochs in range(args.continue_from_epoch):
            learning_rate = lr_reduction_schedule(_previous_epochs, learning_rate) or learning_rate

    if args.model[0] == ":":
        model_name = args.model[1:]
        _is_builtin_model = True
    else:
        print("Continuing training on model", args.model)
        model_name = "mobilenet"
        # Based on the model name we could infer a re-starting epoch
        # TODO infer epoch number of saved model

    # Dense model
    if _is_builtin_model:
        # Is a built-in model = load from keras
        if args.random_weights:
            model = keras.applications.MobileNet(weights=None)
        else:
            model = keras.applications.MobileNet()
    else:
        # The model is not built-in = load from disk
        # Just in case, load our usual custom objects
        c_obj = {'Sparse': Sparse,
                 'SparseConv2D': SparseConv2D,
                 'SparseDepthwiseConv2D': SparseDepthwiseConv2D,
                 'NoisySGD': NoisySGD}
        # Retrieve model from disk
        model = load_model(args.model, custom_objects=c_obj)

    if args.sparse_layers and not args.soft_rewiring:
        if args.conn_decay:
            print("Connectivity decay rewiring enabled", conn_decay_values)
            model = replace_dense_with_sparse(
                model,
                activation=args.activation, batch_size=batch,
                builtin_sparsity=builtin_sparsity,
                reg_coeffs=alphas,
                conn_decay=conn_decay_values, no_cache=args.no_cache,
                random_weights=args.random_weights)
        else:
            model = replace_dense_with_sparse(
                model,
                activation=args.activation, batch_size=batch,
                builtin_sparsity=builtin_sparsity,
                reg_coeffs=alphas, no_cache=args.no_cache,
                random_weights=args.random_weights)
    elif args.sparse_layers and args.soft_rewiring:
        print("Soft rewiring enabled", args.soft_rewiring)
        model = replace_dense_with_sparse(
            model,
            activation=args.activation, batch_size=batch,
            reg_coeffs=alphas, no_cache=args.no_cache,
            random_weights=args.random_weights)

    model.summary()

    dataset_info = load_and_preprocess_dataset(
        'imagenet', batch_size=batch, path=args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
        val_steps_per_epoch=args.val_steps_per_epoch)
    train_gen = dataset_info['train']
    val_gen = dataset_info['val']
    input_shape = dataset_info['input_shape']
    num_classes = dataset_info['num_classes']
    no_train = dataset_info['no_train']
    no_val = dataset_info['no_val']

    # set up steps_per_epoch
    steps_per_epoch = args.steps_per_epoch or no_train // batch
    validation_steps_per_epoch = args.val_steps_per_epoch or no_val // batch

    print("=" * 60)
    print("Quick reports")
    print("-" * 60)
    print("Training Steps per epoch", steps_per_epoch)
    print("Validation Steps per epoch", validation_steps_per_epoch)
    print("Number of classes:", num_classes)
    print("Number of training examples:", no_train)
    print("Number of validation examples:", no_val)
    print("=" * 60)

    activation_name = "relu"
    loss_name = "crossent"

    # Retrieve optimizer and its name (for files and reports)
    optimizer, optimizer_name = extract_optimizer_from_args(learning_rate)

    loss = keras.losses.categorical_crossentropy

    # disable rewiring with sparse layers to see the performance of the layer
    # when 90% of connections are disabled and static
    deep_r = RewiringCallback(fixed_conn=args.disable_rewiring,
                              soft_limit=args.soft_rewiring,
                              asserts_on=args.asserts_on)

    lr_schedule = LearningRateScheduler(lr_reduction_schedule,
                                        verbose=args.verbose)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

    suffix = ""
    if args.suffix:
        suffix = "_" + args.suffix

    sparse_name = generate_sparsity_suffix()
    __filename = generate_filename(
        optimizer_name, activation_name, sparse_name, loss_name, suffix,
        args.random_weights)

    if args.result_filename:
        output_filename = args.result_filename
    else:
        output_filename = "results_for_" + model_name + __filename

    __acr_filename = "models/" + generate_filename(
        optimizer_name, activation_name, sparse_name, loss_name, suffix,
        args.random_weights,
        acronym=True)
    checkpoint_filename = __acr_filename + \
                          "_weights.{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint_callback = ModelCheckpoint(checkpoint_filename, period=5)

    csv_path = os.path.join(args.result_dir, output_filename + ".csv")
    csv_logger = keras.callbacks.CSVLogger(
        csv_path,
        separator=',',
        append=True)

    callback_list = []
    if args.sparse_layers:
        callback_list.append(deep_r)

    if args.tensorboard:
        tb_log_filename = "./sparse_logs" if args.sparse_layers else "./dense_logs"

        tb = keras.callbacks.TensorBoard(
            log_dir=tb_log_filename,
            histogram_freq=0,  # turning this on needs validation_data in model.fit
            batch_size=batch, write_graph=True,
            write_grads=True, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None,
            embeddings_metadata=None, embeddings_data=None,
            update_freq='epoch')
        callback_list.append(tb)

    callback_list.append(csv_logger)
    callback_list.append(checkpoint_callback)
    callback_list.append(lr_schedule)

    # record weight information before learning
    weights_from_model(model)

    if not args.data_augmentation:

        model.fit_generator(train_gen,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            verbose=args.verbose,
                            callbacks=callback_list,
                            validation_data=val_gen,
                            validation_steps=validation_steps_per_epoch,
                            shuffle=True,
                            max_queue_size=10,
                            use_multiprocessing=True,
                            workers=1 if os.name != "nt" else 0,
                            initial_epoch=args.continue_from_epoch
                            )
    else:
        raise NotImplementedError("Data augmentation not currently supported for "
                                  "Mobilenet trained on Imagenet")

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

    score = model.evaluate_generator(val_gen,
                                     steps=validation_steps_per_epoch,
                                     max_queue_size=5,
                                     verbose=args.verbose)
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])


if __name__ == "__main__":
    train_and_test_mobilenet()
