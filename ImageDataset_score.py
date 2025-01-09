import torch
from torch.utils.data import Dataset, DataLoader


class ScoreDataset(Dataset):
    def __init__(self, pt_file="./Dataset/aesthetics_score/train.pt"):
        data = torch.load(pt_file)
        self.scores = data['aesthetics_scores']  # [3162, 5]
        self.filenames = data['filenames']  # [3162, 5]

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # 返回指定索引的数据
        sample = {
            'scores': torch.tensor(self.scores[idx], dtype=torch.float32),  # [5]
            'filenames': self.filenames[idx]  # [5]
        }
        return sample


if __name__ == "__main__":
    dataset = ScoreDataset()
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                              num_workers=8, pin_memory=True)
    print(len(train_loader))

    for batch in train_loader:
        scores = batch['scores']  # [5]
        filenames = batch['filenames']  # [5]
        print("Scores:", scores)
        print("Filenames:", filenames)
        break