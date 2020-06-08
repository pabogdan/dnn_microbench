import ntpath

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from keras_rewiring.experiments.common import *
# network generation imports
from keras_rewiring.experiments.mnist.lenet_300_100_model_setup import \
    generate_lenet_300_100_model, \
    generate_sparse_lenet_300_100_model, \
    convert_model_to_tf


def test_lenet_300_100_using_tf(filename, no_runs):
    print("=" * 80)
    print(i)
    print("-" * 80)
    dataset_info = load_and_preprocess_dataset(
        'mnist', categorical_output=False)
    x_test, y_test = dataset_info['test']
    x_test = x_test.reshape(x_test.shape[0], 1, np.prod(x_test.shape[1:]))
    batch = 10
    # Reinstantiate the model
    c_obj = {'Sparse': Sparse,
             'SparseConv2D': SparseConv2D,
             'SparseDepthwiseConv2D': SparseDepthwiseConv2D,
             'NoisySGD': NoisySGD}
    model = keras.models.load_model(filename, custom_objects=c_obj)

    model = convert_model_to_tf(
        model)

    # Save current model as SavedModel


    print("no runs", no_runs)
    times = np.zeros(no_runs)
    scores = []
    for ni in range(no_runs):
        tb_log_filename = "./tf_inference_logs"
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
        softmaxed_predictions = model(x_test)
        end_time = plt.datetime.datetime.now()
        predictions = np.argmax(np.asarray(softmaxed_predictions), axis=-1)

        # Report accuracy and generate
        # print(classification_report(y_test, predictions))
        total_time = end_time - start_time
        times[ni] = total_time.total_seconds()
        scores.append(accuracy_score(y_test, predictions, normalize=True))
    print("mean time", np.mean(times))
    print("std time", np.std(times))
    base_name_file = str(ntpath.basename(filename))[:-3]
    csv_path = os.path.join(args.result_dir, "tf_" + base_name_file + ".csv")
    np.savetxt(csv_path, times, delimiter=",")

    f = plt.figure(1, figsize=(9, 9), dpi=400)
    plt.hist(times, bins=20,
             rasterized=True,
             edgecolor='k')

    plt.ylabel("Count")
    plt.xlabel("Inference duration (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir,
                             "tf_hist_times_for_" + base_name_file + ".png"))
    plt.close(f)
    return scores, times


if __name__ == "__main__":
    for i in args.model:
        if args.just_test:
            no_runs = 10
            s, t = test_lenet_300_100_using_tf(i, no_runs)
            print("The score are:", s)
