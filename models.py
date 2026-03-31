"""
模型定义：特征提取器
"""
import torch
import torch.nn as nn
import timm
import torchvision.models as models
from transformers import ViTModel, ViTImageProcessor

class FeatureExtractor(nn.Module):
    """特征提取器基类：升级版支持单流与混合双流"""
    
    def __init__(self, model_type='resnet50', feature_dim=512, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.model_type = model_type
        self.feature_dim = feature_dim
        
        if model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.fc = nn.Linear(2048, feature_dim)
            
        elif model_type == 'inception_resnet_v2':
            self.backbone = timm.create_model('inception_resnet_v2', pretrained=pretrained, num_classes=0)
            self.fc = nn.Linear(1536, feature_dim)
            
        elif model_type == 'vit_b16':
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.fc = nn.Linear(768, feature_dim)
            
        # ========================================================
        # 【新增】：CNN-Transformer 混合双流架构
        # ========================================================
        elif model_type == 'hybrid':
            # 分支 1：CNN (专门提取星毛/鳞片纹理)
            resnet = models.resnet50(pretrained=pretrained)
            # 我们只取到 layer3，保留分辨率以抓取微观毛被特征
            self.cnn_branch = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3
            )
            self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.cnn_fc = nn.Linear(1024, feature_dim)
            
            # 分支 2：ViT (专门提取宏观叶型/生境轮廓)
            self.vit_branch = ViTModel.from_pretrained('/data/yutong/models/vit-base-patch16-224')
            self.vit_fc = nn.Linear(768, feature_dim)
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def forward(self, x_global, x_local=None):
        """
        前向传播：同时接收全局图(x_global)和局部图(x_local)
        """
        if self.model_type == 'hybrid':
            # 1. 提取微观纹理特征 (CNN)
            cnn_feat = self.cnn_branch(x_local)
            cnn_feat = torch.flatten(self.cnn_pool(cnn_feat), 1)
            cnn_feat = self.cnn_fc(cnn_feat)
            cnn_feat = nn.functional.normalize(cnn_feat, p=2, dim=1)
            
            # 2. 提取宏观轮廓特征 (ViT)
            vit_outputs = self.vit_branch(pixel_values=x_global)
            vit_feat = vit_outputs.last_hidden_state[:, 0, :]
            vit_feat = self.vit_fc(vit_feat)
            vit_feat = nn.functional.normalize(vit_feat, p=2, dim=1)
            
            # 3. 带权重的特征融合（强迫 AI 关注星状毛）
            # 注意：权重系数在后续将从 config 中动态传入，这里默认0.7
            texture_weight = 0.7 
            fused_feat = (texture_weight * cnn_feat) + ((1.0 - texture_weight) * vit_feat)
            features = fused_feat
            
        elif self.model_type == 'vit_b16':
            outputs = self.backbone(pixel_values=x_global)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0, :]
            features = self.fc(features)
        else:
            features = self.backbone(x_global)
            features = features.view(features.size(0), -1)
            features = self.fc(features)
        
        # 统一进行最后一次 L2 归一化
        features = nn.functional.normalize(features, p=2, dim=1)
        return features

class TripletNetwork(nn.Module):
    """Triplet网络：升级为支持双流输入"""
    def __init__(self, feature_extractor):
        super(TripletNetwork, self).__init__()
        self.feature_extractor = feature_extractor
    
    def forward(self, anchor_g, anchor_l, positive_g, positive_l, negative_g, negative_l):
        # 将 全局流(_g) 和 局部流(_l) 同时喂给特征提取器
        anchor_feat = self.feature_extractor(anchor_g, anchor_l)
        positive_feat = self.feature_extractor(positive_g, positive_l)
        negative_feat = self.feature_extractor(negative_g, negative_l)
        
        return anchor_feat, positive_feat, negative_feat
class ClassificationHead(nn.Module):
    """分类头：用于分类任务"""
    
    def __init__(self, feature_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, features):
        return self.classifier(features)

def create_model(model_type='resnet50', feature_dim=512, num_classes=None, 
                pretrained=True, use_triplet=True):
    """
    创建模型
    
    Args:
        model_type: 模型类型
        feature_dim: 特征维度
        num_classes: 类别数（如果为None，则只创建特征提取器）
        pretrained: 是否使用预训练权重
        use_triplet: 是否用于Triplet Loss
    
    Returns:
        model
    """
    feature_extractor = FeatureExtractor(model_type, feature_dim, pretrained)
    
    if use_triplet:
        model = TripletNetwork(feature_extractor)
    elif num_classes is not None:
        model = nn.Sequential(
            feature_extractor,
            ClassificationHead(feature_dim, num_classes)
        )
    else:
        model = feature_extractor
    
    return model

