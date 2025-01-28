import argparse
import os
import pickle as pkl

from cluster import Clusters
from config import Config

# Login into huggingface_hub
from huggingface_hub import login
from trainer import Trainer
from utils import autotune_batch_size, prepare_hub_name

os.environ["HUGGINGFACE_TOKEN"] = XXXX
USER = XXXX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument(
        "--enable_BSGC", action="store_true", help="Enable BSGC when flag is present"
    )
    parser.add_argument(
        "--do_modularity",
        action="store_true",
        help="Enable modularity when flag is present",
    )
    parser.add_argument(
        "--mix_data", action="store_true", help="Train on both RAVEL and WIKI"
    )

    args = parser.parse_args()

    # Validate BSGC and modularity dependency
    if args.enable_BSGC and not args.do_modularity:
        parser.error("--enable_BSGC requires --do_modularity to be enabled")

    print(f"All arguments: {parser.parse_args()}")

    args = parser.parse_args()
    clusters_list = [4]

    # Login to huggingface
    login(token=os.environ["HUGGINGFACE_TOKEN"])

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
        model, tokenizer = Config(
            model_name=args.model_name
        ).forward()  # Get fresh model instance
        clusters = Clusters(model, tokenizer, num_clusters, args.enable_BSGC)
        bs = autotune_batch_size(
            model, tokenizer
        )  # Check what this does in utils: I set a ceiling at bs=32 as to avoid overfitting given simplicity
        args.batch_size = bs

        # CLUSTER THE MODEL WEIGHTS AND GET (U, V) FOR EACH LAYER
        svd_dict = (
            clusters.forward()
        )  # This is the dictionary containing U, V for each layer for a given k!
        svd_clusters_dict[num_clusters] = svd_dict  # Storing for dumping later

        repo_id = prepare_hub_name(args, num_clusters, USER) + "FixCluster"

        # INITIALIZE TRAINER
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_clusters=num_clusters,
            steps_to_cluster=150,  # Play with this value to start clustering at a later moment!
            model_name=args.model_name.split("/")[-1] + "_lr_" + str(args.lr),
            enable_BSGC=args.enable_BSGC,
            do_modularity=args.do_modularity,
            mix_ravel=args.mix_data,
            ckpt_name=repo_id,
        )

        # TRAIN THE MODEL WITH CLUSTERED WEIGHTS
        cluster_metrics = trainer.train(
            cluster_dict=svd_dict,
            num_epochs=args.num_epochs,
            lr=args.lr,
        )

        all_clusters_metrics[num_clusters] = cluster_metrics

        # Push model, tokenizer to huggingface
        trainer.model.push_to_hub(repo_id)
        trainer.tokenizer.push_to_hub(repo_id)

    # DUMP THE SVD DICTIONARY
    with open(f"svd_dict_{args.model_name}_{args.lr}.pkl", "wb") as f:
        pkl.dump(svd_clusters_dict, f)

    # DUMP THE METRICS DICTIONARY
    with open(f"metrics_dict{args.model_name}_{args.lr}.pkl", "wb") as f:
        pkl.dump(all_clusters_metrics, f)


if __name__ == "__main__":
    main()
