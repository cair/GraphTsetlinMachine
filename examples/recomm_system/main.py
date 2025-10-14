from datetime import datetime
import graph_nn
import graph_tm
import tm_classifier

dataset_noise_ratios = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
num_iterations = 10
exp_id = datetime.now().strftime("%Y%m%d%H%M%S")

print(f"{datetime.now()}, Setup the environment ...")
print(f"Experiment ID: {exp_id}")

for i in range(1, num_iterations + 1):
    print(f"Iteration {i} of {num_iterations}")

    for N in dataset_noise_ratios:
        print(f"{datetime.now()}, Running Graph NN ...")
        args_nn = graph_nn.default_args(
            dataset_noise_ratio=N,
            exp_id=exp_id
        )
        graph_nn.main(args_nn)

        print(f"{datetime.now()}, Running Graph Tsetlin Machine ...")
        args_tm = graph_tm.default_args(
            dataset_noise_ratio=N,
            exp_id=exp_id
        )
        graph_tm.main(args_tm)

        print(f"{datetime.now()}, Running Tsetlin Machine Classifier ...")
        args_classifier = tm_classifier.default_args(
            dataset_noise_ratio=N,
            exp_id=exp_id
        )
        tm_classifier.main(args_classifier)