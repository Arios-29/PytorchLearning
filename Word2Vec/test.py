from torch.utils.data import DataLoader

from Word2Vec.Loader import PTBLoader, negative_sampling
from Word2Vec.PTBDataset import PTBDataset, read_a_batch

if __name__ == "__main__":

    loader = PTBLoader()
    centers, contexts = loader.get_centers_and_contexts(5)
    negatives = negative_sampling(contexts, loader.get_sampling_weights(), 5)

    ptb_dataset = PTBDataset(centers, contexts, negatives)
    data_iter = DataLoader(ptb_dataset, batch_size=512, shuffle=True, collate_fn=read_a_batch, num_workers=4)
    for batch in data_iter:
        for name, data in zip(["centers", "context_negatives", "masks", "labels"], batch):
            print(name, "shape:", data.size())
        break
