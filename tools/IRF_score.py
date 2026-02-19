from torch import nn
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', s=' + str(self.s) \
            + ', m=' + str(self.m) + ')'


class add_layer_model(nn.Module):
    def __init__(self, backbone):
        super(add_layer_model, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(1024,64)
        #self.bn = nn.BatchNorm1d(64)
        self.fc2 = AddMarginProduct(64, 10000, s=30, m=0.65)
        self.drpout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.backbone(x)
        # x = self.drpout(x)
        # x = self.fc1(x)
        # x = self.bn(x)
        # x = self.fc2(x, label)
        return x


def init_IRF_model(IRF_model_path, vit_model, device):

    backbone, _, preprocess = open_clip.create_model_and_transforms(vit_model)
    backbone = backbone.visual
    model = add_layer_model(backbone)
    weight_try = torch.load(IRF_model_path)
    weight_clear = {}
    for i in weight_try.items():
        weight_clear[i[0].split('module.')[-1]] = i[1]
    weight_try.popitem('fc2.weight')
    positional_embedding = weight_clear['backbone.positional_embedding']
    pos_embed_before = positional_embedding[:1,:]
    pos_embed_after = positional_embedding[1:,:]
    pos_embed_after = pos_embed_after.view(1, 24, 24, 1280).permute(0, 3, 1, 2)
    pos_embed_after = torch.nn.functional.interpolate(pos_embed_after, size=(16,16), mode='bicubic')
    pos_embed_after = pos_embed_after.permute(0, 2, 3, 1).view(16 * 16, 1280)
    pos_embed = torch.cat([pos_embed_before, pos_embed_after])
    weight_clear['backbone.positional_embedding'] = pos_embed
    model.load_state_dict(weight_clear , strict=True)
    model.eval()
    model.to(device)
    return model, preprocess


def calculate_IRF_score(model, preprocess, image1, image2, device):
    with torch.no_grad():
        image1 = preprocess(image1).unsqueeze(0).to(device)
        image2 = preprocess(image2).unsqueeze(0).to(device)
        feature_map1 = model(image1)
        feature_map2 = model(image2)
        # print(feature_map1.dtype)
        similarity = torch.nn.functional.cosine_similarity(feature_map1, feature_map2).item()
        # print(f"Cosine similarity between the two images: {similarity:.4f}")
        # feature_map1 /= feature_map1.norm(dim=-1, keepdim=True)
        # feature_map2 /= feature_map2.norm(dim=-1, keepdim=True)
        # similarity = feature_map1.cpu().numpy() @ feature_map2.cpu().numpy().T
        # print(f"Cosine similarity between the two images: {similarity}")
        return similarity
