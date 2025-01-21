from utils import spectral_clustering


class Clusters:
    def __init__(self, model, tokenizer, num_clusters):
        self.model = model
        self.tokenizer = tokenizer
        self.num_clusters = num_clusters
        self.num_layers = self.model.config.num_hidden_layers

    def forward(self):
        svd_dict = {}
        for layer_idx in range(self.num_layers):
            U, V = spectral_clustering(
                self.model.gpt_neox.layers[
                    layer_idx
                ].mlp.dense_h_to_4h.weight,  # This is pythia-specific, ideally should be abstracted
                self.num_clusters,
            )
            svd_dict[layer_idx] = (U, V)

        return svd_dict
