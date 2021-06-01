import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import re
from clinicgen.models.image import ImageClassification


class MLClassifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        # self.backbone = nn.Sequential(*list(densenet.children())[:-1])
        self.densenet121.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, img1, img2):
        x1 = self.densenet121.features(img1)
        x1 = F.relu(x1, inplace=True)
        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x1 = torch.flatten(x1, 1)
        x1 = self.densenet121.classifier(x1)
        x2 = self.densenet121.features(img2)
        x2 = F.relu(x2, inplace=True)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x2 = torch.flatten(x2, 1)
        x2 = self.densenet121.classifier(x2)
        return x1 + x2


class ClsAttention(nn.Module):

    def __init__(self, feat_size, num_classes):
        super().__init__()
        self.feat_size = feat_size
        self.num_classes = num_classes
        self.channel_w = nn.Conv2d(feat_size, num_classes, 1, bias=False)

    def forward(self, feats):
        # feats: batch size x feat size x H x W
        batch_size, feat_size, H, W = feats.size()
        att_maps = self.channel_w(feats)
        att_maps = torch.softmax(att_maps.view(batch_size, self.num_classes, -1), dim=2)
        feats_t = feats.view(batch_size, feat_size, H * W).permute(0, 2, 1)
        cls_feats = torch.bmm(att_maps, feats_t)
        return cls_feats


class GCLayer(nn.Module):

    def __init__(self, in_size, state_size):
        super().__init__()
        self.condense = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.condense_norm = nn.BatchNorm1d(state_size)
        self.fw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.fw_norm = nn.BatchNorm1d(state_size)
        self.bw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.bw_norm = nn.BatchNorm1d(state_size)
        self.update = nn.Conv1d(3 * state_size, in_size, 1, bias=False)
        self.update_norm = nn.BatchNorm1d(in_size)
        self.relu = nn.ReLU(inplace=True)
        # v2:
        self.dropout = nn.Dropout(0.5)

    def forward(self, states, fw_A, bw_A):
        # states: batch size x feat size x nodes
        condensed = self.relu(self.condense_norm(self.condense(states)))
        fw_msg = self.relu(self.fw_norm(self.fw_trans(states).bmm(fw_A)))
        bw_msg = self.relu(self.bw_norm(self.bw_trans(states).bmm(bw_A)))
        updated = self.update_norm(self.update(torch.cat((condensed, fw_msg, bw_msg), dim=1)))
        updated = self.relu(self.dropout(updated) + states)
        return updated


class GCN(nn.Module):

    def __init__(self, in_size, state_size, steps=3):
        super().__init__()
        self.in_size = in_size
        self.state_size = state_size
        self.steps = steps

        # layers = []
        # for istep in range(steps):
        #     layers.append(GCLayer(in_size, state_size))
        # self.layers = nn.Sequential(*layers)
        self.layer1 = GCLayer(in_size, state_size)
        self.layer2 = GCLayer(in_size, state_size)
        self.layer3 = GCLayer(in_size, state_size)

    def forward(self, states, fw_A, bw_A):
        states = states.permute(0, 2, 1)
        states = self.layer1(states, fw_A, bw_A)
        states = self.layer2(states, fw_A, bw_A)
        # states = self.layer3(states, fw_A, bw_A)
        return states.permute(0, 2, 1)


class GCNClassifier(nn.Module):

    def __init__(self,kg_dir=None,image_model='densenet', image_pretrained=None, finetune_image=False):
        super().__init__()
        # load knowledge graph
        kg = []
        with open(kg_dir) as f:
            for line in f.readlines():
                kg.append([float(i) for i in re.split(',|\[|\]', line) if i not in ['', ' ', '\n']])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        fw_adj = torch.tensor(kg, dtype=torch.float, device=device)
        bw_adj = fw_adj.t()

        self.num_classes = fw_adj.shape[0]-1  #consider there is a global feature
        self.densenet121 = models.densenet121(pretrained=True)
        feat_size = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(feat_size, self.num_classes)
        self.cls_atten = ClsAttention(feat_size, self.num_classes)
        self.gcn = GCN(feat_size, 256)
        # v1:
        self.fcs = nn.ModuleList([nn.Linear(feat_size, 1) for _ in range(self.num_classes)])
        # v2:
        self.fc2 = nn.Linear(feat_size, self.num_classes)

        fw_D = torch.diag_embed(fw_adj.sum(dim=1))
        bw_D = torch.diag_embed(bw_adj.sum(dim=1))
        inv_sqrt_fw_D = fw_D.pow(-0.5)
        inv_sqrt_fw_D[torch.isinf(inv_sqrt_fw_D)] = 0
        inv_sqrt_bw_D = bw_D.pow(-0.5)
        inv_sqrt_bw_D[torch.isinf(inv_sqrt_bw_D)] = 0
        self.fw_A = inv_sqrt_fw_D.mm(fw_adj).mm(inv_sqrt_bw_D)
        self.bw_A = inv_sqrt_bw_D.mm(bw_adj).mm(inv_sqrt_fw_D)
        # self.image_feats, image_dim = ImageClassification.image_features(image_model, not finetune_image, True,
        #                                                                  image_pretrained, device)

    def forward(self, img):
        batch_size = img.size(0)
        fw_A = self.fw_A.repeat(batch_size, 1, 1)
        bw_A = self.bw_A.repeat(batch_size, 1, 1)
        cnn_feats = self.densenet121.features(img)
        cnn_feats_deal = cnn_feats.flatten(start_dim=-2, end_dim=-1)
        cnn_feats_deal = cnn_feats_deal.permute(0, 2, 1)

        global_feats = cnn_feats.mean(dim=(2, 3))
        cls_feats = self.cls_atten(cnn_feats)
        node_feats = torch.cat((global_feats.unsqueeze(1), cls_feats), dim=1)
        node_states = self.gcn(node_feats, fw_A, bw_A)
        global_states = node_states.mean(dim=1)
        # logits = self.fc2(global_states)
        return cnn_feats_deal,node_feats, node_states,global_states
