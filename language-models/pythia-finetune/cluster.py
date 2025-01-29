from utils import get_mlp_parameters, spectral_clustering


class Clusters:
    def __init__(self, model, tokenizer, num_clusters, enable_BSGC):
        self.model = model
        self.tokenizer = tokenizer
        self.num_clusters = num_clusters
        self.num_layers = self.model.config.num_hidden_layers
        self.BSGC = enable_BSGC

    def forward(self):
        svd_dict = {}
        blocks_to_cluster = get_mlp_parameters(self.model, "in")
        for layer_idx in range(self.num_layers):
            U, V = (
                spectral_clustering(
                    blocks_to_cluster[layer_idx],
                    self.num_clusters,
                )
                if self.BSGC
                else (None, None)
            )
            svd_dict[layer_idx] = (U, V)

        return svd_dict
