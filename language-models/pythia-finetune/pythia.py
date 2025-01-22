import argparse
import pickle as pkl

from cluster import Clusters
from config import Config
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    args = parser.parse_args()
    clusters_list = [2, 4, 5, 6, 8]

    svd_clusters_dict = {
        cluster_idx: {} for cluster_idx in clusters_list
    }  # contains (U, V for each layer) for each cluster

    # Dictionary to store metrics per cluster
    all_clusters_metrics = {
        num_clusters: {"train_losses": [], "cluster_losses": [], "val_losses": []}
        for num_clusters in clusters_list
    }

    for num_clusters in clusters_list:  # Loop over possible k
        # Get fresh model instance for each cluster
        print(f"Training for cluster {num_clusters}")

        # INITIALIZE MODEL
        model, tokenizer = Config().forward()  # Get fresh model instance
        clusters = Clusters(model, tokenizer, num_clusters)

        # CLUSTER THE MODEL WEIGHTS AND GET (U, V) FOR EACH LAYER
        svd_dict = (
            clusters.forward()
        )  # This is the dictionary containing U, V for each layer for a given k!
        svd_clusters_dict[num_clusters] = svd_dict  # Storing for dumping later

        # INITIALIZE TRAINER
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_clusters=num_clusters,
        )

        # TRAIN THE MODEL WITH CLUSTERED WEIGHTS
        cluster_metrics = trainer.train(
            cluster_dict=svd_dict,
            num_epochs=args.num_epochs,
            lr=args.lr,
        )

        all_clusters_metrics[num_clusters] = cluster_metrics

    # DUMP THE SVD DICTIONARY
    with open("svd_dict.pkl", "wb") as f:
        pkl.dump(svd_clusters_dict, f)

    # DUMP THE METRICS DICTIONARY
    with open("metrics_dict.pkl", "wb") as f:
        pkl.dump(all_clusters_metrics, f)


if __name__ == "__main__":
    main()

# def main():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
#     parser.add_argument("--batch_size", type=int, default=128)
#     args = parser.parse_args()
#     c = Config()
#     model, tokenizer, num_clusters = c.forward()
#     trainer = Trainer(model, tokenizer, num_clusters, batch_size=args.batch_size)

#     # Try to load existing SVD dictionary, if not exists, create new one
#     if os.path.exists("svd_dict.pkl"):
#         with open("svd_dict.pkl", "rb") as f:
#             svd_dict = pkl.load(f)
#     else:
#         clusters = Clusters(model, tokenizer, num_clusters)
#         svd_dict = clusters.forward()

#     for (
#         cluster_value,
#         cluster_dict,
#     ) in svd_dict.items():  # Loop through num_clusters (keys of svd_dict) noqa
#         # for key, value in svd_dict[cluster_value].items():
#         #     # pprint(value)
#         #     U, V = value
#         # break
#         print(f"C: {cluster_value}")
#         print(
#             f"CD: {cluster_dict.keys()}"
#         )  # cluster_dict.keys() is a list of layer indices, each containing U, V indices pair for that layer
#         print("---" * 10)
#         # cluster_dict[0] == tuple(U, V)

#         trainer.train(cluster_dict)
