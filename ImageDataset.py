import torch
from torch.utils.data import Dataset, DataLoader
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): The root directory containing .pt files for the dataset.
        """
        self.file_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.pt')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (artifact_map, misalignment_map, scores, token_label)
        """
        data = torch.load(self.file_paths[idx])

        filename = data['filename']
        artifact_map = data['artifact_map']
        misalignment_map = data['misalignment_map']
        # scores = data['scores']  # [aesthetics_score, artifact_score, misalignment_score, overall_score]
        scores = {
            'aesthetics_score': data['scores'][0],
            'artifact_score': data['scores'][1],
            'misalignment_score': data['scores'][2],
            'overall_score': data['scores'][3]
        }
        token_label = data['token_label']

        # return artifact_map, misalignment_map, scores, token_label
        return {'filename': filename,
                'artifact_map': artifact_map,
                'misalignment_map': misalignment_map,
                'scores': scores,
                'token_label': token_label}

if __name__ == "__main__":
    # Define paths to the processed datasets
    train_dir = "./Data_richhf18k/torch/train"
    dev_dir = "./Data_richhf18k/torch/dev"
    test_dir = "./Data_richhf18k/torch/test"

    # Create dataset objects
    train_dataset = ImageDataset(train_dir)
    dev_dataset = ImageDataset(dev_dir)
    test_dataset = ImageDataset(test_dir)

    # Define data loaders
    bs = 2
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8)
    dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)

    val_count = 0
    # Iterate through train_loader to test
    for batch in train_loader:
        # artifact_maps, misalignment_maps, scores, token_labels = batch
        filenames = batch['filename']
        artifact_maps = batch['artifact_map']
        misalignment_maps = batch['misalignment_map']
        scores = batch['scores']
        token_labels = batch['token_label']

        # print([filename.split('/')[0] for filename in filenames])
        # tmp = [filename.split('/')[0] for filename in filenames]
        # if any(split != "train" and split != "test" for split in tmp):
        #     val_count += 1

        print("Filenames:", filenames)
        # print("Artifact maps shape:", artifact_maps.shape)  # [batch_size, H, W]
        # print("Misalignment maps shape:", misalignment_maps.shape)  # [batch_size, H, W]
        # print("Scores:", scores)  # dict
        # print("Token labels:", token_labels)  # List of strings

        # import matplotlib.pyplot as plt
        #
        # batch_idx = 0  # Visualize the first image in the batch
        # artifact_map = artifact_maps[batch_idx]  # Shape: [channels, height, width] or [height, width]
        # # misalignment_map = misalignment_maps[batch_idx]
        # num_channels = artifact_map.shape[0] if artifact_map.ndimension() == 3 else 1
        # fig, axes = plt.subplots(1, num_channels + 1, figsize=(5 * (num_channels + 1), 5))
        #
        # # Add a large title for the heatmap
        # fig.suptitle("Artifact Heatmap", fontsize=20, y=1.05)
        #
        # # Display each channel of the artifact map
        # if num_channels > 1:  # Multi-channel case
        #     for ch in range(num_channels):
        #         ax = axes[ch]
        #         ax.imshow(artifact_map[ch], cmap='hot')
        #         ax.set_title(f"Channel {ch + 1}", fontsize=14)
        #         ax.axis('off')
        # else:  # Single-channel case
        #     axes[0].imshow(artifact_map)
        #     axes[0].set_title("Channel 1", fontsize=14)
        #     axes[0].axis('off')
        #
        # # Add the original image (e.g., misalignment map or other raw input) at the end
        # ax_orig = axes[-1]
        # ax_orig.imshow(artifact_map)  # Assuming single-channel grayscale image
        # ax_orig.set_title("Original Image", fontsize=14)
        # ax_orig.axis('off')
        #
        # # Adjust layout and display
        # plt.tight_layout()
        # plt.show()

        break

    # print(val_count)
