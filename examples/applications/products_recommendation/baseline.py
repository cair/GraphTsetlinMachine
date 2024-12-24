import logging
import argparse
import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import prepare_dataset
from tmu.data import MNIST
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        accuracy=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )

def prepare_data():
    # Step 1: Load and encode dataset
    data = prepare_dataset.aug_amazon_products()
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    le_category = LabelEncoder()
    le_rating = LabelEncoder() 
    data['user_id'] = le_user.fit_transform(data['user_id'])
    data['product_id'] = le_item.fit_transform(data['product_id'])
    data['category'] = le_category.fit_transform(data['category'])
    data['rating'] = le_rating.fit_transform(data['rating'])
    
    x = data[['user_id', 'product_id', 'category']].values  
    y = data['rating'].values 
    # Step 3: One-hot encode features
    encoder = OneHotEncoder(sparse_output=False, dtype=np.uint32)  
    x_binary = encoder.fit_transform(x)

    # Verify feature dimensions
    print(f"Number of features after one-hot encoding: {x_binary.shape[1]}")

    x_train, x_test, y_train, y_test = train_test_split(x_binary, y, test_size=0.2, random_state=42)

    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)
    
    print("x_train shape:", x_train.shape, "dtype:", x_train.dtype)
    print("y_train shape:", y_train.shape, "dtype:", y_train.dtype)
    print("x_test shape:", x_test.shape, "dtype:", x_test.dtype)
    print("y_test shape:", y_test.shape, "dtype:", y_test.dtype)

    return x_train, x_test, y_train, y_test

def main(args):   
    experiment_results = metrics(args)
    X_train, X_test, Y_train, Y_test = prepare_data()
    
    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses
    )
    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                res = tm.fit(
                    X_train,
                    Y_train,
                )

            experiment_results["train_time"].append(benchmark1.elapsed())
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(X_test) == Y_test).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

        if args.platform == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)

    return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=5000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--platform", default="CPU_sparse", type=str, choices=["CPU", "CPU_sparse", "CUDA"])
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=60, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    results = main(default_args())
    _LOGGER.info(results)