from transformer_lens import HookedTransformer

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformer_lens.evals import evaluate

base_model = HookedTransformer.from_pretrained("gpt2-small")
base_model.to(device)

import torch as t

def clusterability(matrix, cluster_U_indices, cluster_V_indices):
    num_clusters = len(cluster_U_indices)
    A = matrix ** 2
    mask = t.zeros_like(A, dtype=t.bool)
    
    for cluster_idx in range(num_clusters):
        u_indices = t.tensor(cluster_U_indices[cluster_idx], dtype=t.long)
        v_indices = t.tensor(cluster_V_indices[cluster_idx], dtype=t.long)
        mask[u_indices.unsqueeze(1), v_indices] = True
    
    intra_cluster_out_sum = t.sum(A[mask])
    total_out_sum = t.sum(A)
    
    return intra_cluster_out_sum / total_out_sum

from transformer_lens.evals import make_wiki_data_loader, make_pile_data_loader, make_owt_data_loader, make_code_data_loader

datasets = {
    'wiki': make_wiki_data_loader(base_model.tokenizer, batch_size=8),
    'pile': make_pile_data_loader(base_model.tokenizer, batch_size=8),
    'owt': make_owt_data_loader(base_model.tokenizer, batch_size=8),
    'code': make_code_data_loader(base_model.tokenizer, batch_size=8),
}

## Expt 1: Wiki on all layer MLP_in

cluster_losses = []
train_losses = []
test_losses = []
lomda = 40.0
num_clusters = 4
block = base_model.blocks[5].mlp.W_in
cluster_size = (block.shape[0] // num_clusters, block.shape[1] // num_clusters)
path = './checkpoints/'
num_epochs = 30
cluster_U_indices = {i: list(range(i*cluster_size[0], (i+1)*cluster_size[0])) for i in range(num_clusters)}
cluster_V_indices = {i: list(range(i*cluster_size[1], (i+1)*cluster_size[1])) for i in range(num_clusters)}
block_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

for block_idx in block_indices:
    model = HookedTransformer.from_pretrained("gpt2-small")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(num_epochs):
        for idx, batch in enumerate(datasets['wiki']):
            tokens = batch['tokens'].to(device)
            cluster_loss_mlp_in = clusterability(model.blocks[block_idx].mlp.W_in, cluster_U_indices, cluster_V_indices)
            train_loss = model(tokens, return_type="loss")
            cluster_loss = cluster_loss_mlp_in
            cluster_losses.append(cluster_loss.item())
            loss = train_loss - lomda * cluster_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {idx}, Train Loss: {train_loss.item()}, Cluster Loss: {cluster_loss.item()}')
            del tokens, train_loss, cluster_loss, loss
            torch.cuda.empty_cache()
    # save the model
    torch.save(model.state_dict(), path + f'wiki_mlp_in_{block_idx}.pt')
    torch.cuda.empty_cache()
    del model, optimizer