import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import models
from transformers import BertModel, BertTokenizer, ViTModel, SwinModel

from ImageDataset import ImageDataset
from ImageDataset_score import ScoreDataset
from utils import load_image_and_prompt


class ScorePredictor(nn.Module):
    def __init__(self):
        super(ScorePredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=384, kernel_size=3, stride=1),
            nn.LayerNorm([384, 12, 12]),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=64, kernel_size=3, stride=1),
            nn.LayerNorm([64, 10, 10]),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=6400, out_features=2048),
            nn.ReLU(),

            nn.Linear(in_features=2048, out_features=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize Conv2d layers with He initialization (Kaiming Normal)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

            # Initialize Linear layers with Xavier initialization (Glorot Normal)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

            # Optionally, for LayerNorm, you can also initialize its weights and biases
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        output = self.fc_layers(x)

        return output


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, image_tokens, text_tokens=None, text_attention_mask=None):
        '''
        :param image_tokens: [5, 197, 768]
        :param text_tokens: [5, 6, 768]
        :param text_attention_mask: [5, 6]
        :return: image_tokens_out.shape [5, 197, 768]
        '''
        image_attention_mask = torch.ones(image_tokens.size(0), image_tokens.size(1), device=image_tokens.device)
        if text_tokens is not None and text_attention_mask is not None:
            combined_attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1)
            combined_tokens = torch.cat([image_tokens, text_tokens], dim=1)  # 连接图像 token 和文本 token
        else:
            combined_attention_mask = image_attention_mask
            combined_tokens = image_tokens

        bert_output = self.bert(inputs_embeds=combined_tokens, attention_mask=combined_attention_mask)

        output = bert_output.last_hidden_state
        image_tokens_out = output[:, :image_tokens.size(1), :]

        return image_tokens_out


class Drew(nn.Module):
    def __init__(self):
        super(Drew, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #
        # self.fusion = SelfAttention()

        self.score_predictor = ScorePredictor()

    def forward(self, images, prompts=None, mode="aesthetics_score"):  # [5, 3, 224, 224]  [5, seq_len]
        vit_outputs = self.vit(pixel_values=images)
        image_tokens = vit_outputs.last_hidden_state  # (5, 197, 768)

        # if prompts is not None:
        #     inputs = self.tokenizer(prompts, padding=True, truncation=True,
        #                             return_tensors='pt').to(images.device)
        #     input_ids = inputs['input_ids']  # [batch_size, text_seq_len]
        #     attention_mask = inputs['attention_mask']  # [batch_size, text_seq_len]
        #     text_outputs = self.text_encoder(input_ids=input_ids,
        #                                      attention_mask=attention_mask)
        #     text_tokens = text_outputs.last_hidden_state  # (batch_size, text_seq_len, 768)
        #
        #     fused_image_tokens = self.fusion(image_tokens,
        #                                      text_tokens, attention_mask)
        # else:
        #     # fused_image_tokens = self.fusion(image_tokens)
        fused_image_tokens = image_tokens

        fused_image_tokens = fused_image_tokens[:, 1:, :]  # remove[cls]
        batch_size, seq_len, d_model = fused_image_tokens.size()
        patch_size, image_size = 16, 224
        num_patches_per_side = image_size // patch_size
        feature_maps = fused_image_tokens.view(batch_size, num_patches_per_side,
                                               num_patches_per_side, d_model)
        feature_maps = feature_maps.permute(0, 3, 1, 2)  # (5, 768, H=14, W=14)

        if mode == "aesthetics_score":
            score_output = self.score_predictor(feature_maps)
            return score_output.squeeze()
        elif mode == "misalignment_score":
            pass


class ScorePredictor2(nn.Module):
    def __init__(self):
        super(ScorePredictor2, self).__init__()

        # feature maps shape [5, 1024, 7, 7]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([512, 7, 7]),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([128, 7, 7]),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 7, 7]),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([16, 7, 7]),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=16 * 7 * 7, out_features=512),  # 784 -> 512
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
            # 使用tanh
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize Conv2d layers with He initialization (Kaiming Normal)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

            # Initialize Linear layers with Xavier initialization (Glorot Normal)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

            # Optionally, for LayerNorm, you can also initialize its weights and biases
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        # self.conv_layers[0] = nn.Conv2d(in_channels=x.shape[1], out_channels=512, kernel_size=3, stride=1, padding=1)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        output = self.fc_layers(x)

        # output = (torch.tanh(output) + 1) / 2  # Use tanh to map to [0, 1]

        return output


class Drew2(nn.Module):
    def __init__(self):
        super(Drew2, self).__init__()
        # self.swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        self.swin = SwinModel.from_pretrained("microsoft/swin-large-patch4-window7-224")

        # self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # self.fusion = SelfAttention()

        self.score_predictor = ScorePredictor2()

    def forward(self, images, prompts=None, mode="aesthetics_score"):  # [5, 3, 224, 224]  [5, seq_len]
        swin_outputs = self.swin(pixel_values=images)
        image_tokens = swin_outputs.last_hidden_state  # (5, 49, 1024)  [batch_size, num_patches, hidden_size]

        fused_image_tokens = image_tokens

        # fused_image_tokens = fused_image_tokens[:, 1:, :]  # remove [CLS]
        batch_size, token_len, d_model = fused_image_tokens.size()  # [5, 49, 1024]
        patch_size = 7  # Swin default patch = 7
        feature_maps = fused_image_tokens.view(batch_size, patch_size,
                                               patch_size, d_model)
        feature_maps = feature_maps.permute(0, 3, 1, 2)  # (5, 1024, H=7, W=7)

        if mode == "aesthetics_score":
            score_output = self.score_predictor(feature_maps)
            return score_output.squeeze()
        elif mode == "misalignment_score":
            pass


class Drew3(nn.Module):
    def __init__(self):
        super(Drew3, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 1)

    def forward(self, image):
        score = self.resnet50(image)
        return score


if __name__ == "__main__":
    train_dir = "./Data_richhf18k/torch/train"
    dev_dir = "./Data_richhf18k/torch/dev"
    test_dir = "./Data_richhf18k/torch/test"
    train_dataset = ScoreDataset()
    dev_dataset = ImageDataset(dev_dir)
    test_dataset = ImageDataset(test_dir)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Drew().to(device)

    for batch in train_loader:
        filenames = batch['filenames']
        artifact_maps = batch['artifact_map']
        misalignment_maps = batch['misalignment_map']
        scores = batch['scores']
        token_labels = batch['token_label']
        images, prompts, artifact_maps, misalignment_maps = load_image_and_prompt(filenames, artifact_maps,
                                                                                  misalignment_maps)

        images = torch.stack(images, dim=0).to(device)  # (batch_size, C, H, W)
        artifact_maps = torch.stack(artifact_maps, dim=0).to(device)  # (batch_size, 1, H, W)
        misalignment_maps = torch.stack(misalignment_maps, dim=0).to(device)  # (batch_size, 1, H, W)

        score_output = model(images, prompts, "aesthetics_score")
        print(score_output)

        break
