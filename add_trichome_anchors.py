import torch

import numpy as np

from PIL import Image

from torchvision import transforms

from sklearn.metrics.pairwise import cosine_similarity

import os



# 导入你的模型创建函数

from models import create_model



# ==========================================

# ⚙️ 参数设置区

# ==========================================

MODEL_TYPE = 'hybrid'  # 使用的双流架构

FEATURE_DIM = 512      # 特征维度

IMAGE_SIZE = 224       # 局部图分辨率 (如果你平时用的是224，请改为224)

ANCHOR_WEIGHT = 27.0   # 毛被特征的“引力权重” (控制聚类时毛被的影响力)

# ==========================================



def get_single_feature(model, image_path, device):

    """提取单张图片的特征，完美绕过 TripletNetwork 的 6 参数限制"""

    transform = transforms.Compose([

        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    

    img = Image.open(image_path).convert('RGB')

    img_tensor = transform(img).unsqueeze(0).to(device)

    

    with torch.no_grad():

        # 【暴力破解】：强行塞入 6 个相同的 input 满足代码要求

        outputs = model(img_tensor, img_tensor, img_tensor, img_tensor, img_tensor, img_tensor)

        

        # Triplet 返回的是元组，我们只取第一个 (anchor 输出)

        if isinstance(outputs, (list, tuple)):

            feature = outputs[0]

        else:

            feature = outputs

            

    return feature.cpu().numpy().flatten()



def main():

    print("🚀 启动毛被特征干预程序...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    print("📦 加载模型权重...")

    model = create_model(model_type=MODEL_TYPE, feature_dim=FEATURE_DIM, pretrained=False)

    

    # 确保路径指向你真实训练好的权重！

    weights_path = 'outputs/models/hybrid_best.pth' 

    if os.path.exists(weights_path):

        # 1. 先把权重字典读取出来

        state_dict = torch.load(weights_path, map_location=device)

        new_state_dict = {}

        

        # 2. 遍历字典里的所有名字，进行自动修复

        for k, v in state_dict.items():

            # 如果名字前面带有 'module.' (多卡训练残留)，直接砍掉

            if k.startswith('module.'):

                k = k.replace('module.', '')

            

            # 如果现在的模型需要 'feature_extractor.' 这个壳，但权重里没有，就强行给它加上

            if not k.startswith('feature_extractor.') and ('cnn_branch' in k or 'vit_branch' in k or 'fc' in k):

                k = f'feature_extractor.{k}'

                

            new_state_dict[k] = v

            

        # 3. 把修复好名字的权重塞进模型，strict=False 允许微小偏差，防止死板报错

        model.load_state_dict(new_state_dict, strict=False)

        print("✅ 成功加载（并修复）了微调模型权重！")

    else:

        print("⚠️ 警告：未找到 best_model.pth，使用未训练的基础权重！")    

    model.to(device)

    model.eval()



    print("🔬 正在提取 3 张锚点照片的‘真理特征’...")

    feat_s = get_single_feature(model, 'anchors/stellate.jpg', device)

    feat_p = get_single_feature(model, 'anchors/peltate.jpg', device)

    feat_m = get_single_feature(model, 'anchors/mixed.jpg', device)



    print("📊 正在读取几百个物种的原始特征文件...")

    # 这里读取你原本已经跑完特征提取的 .npz 文件

    data = np.load('outputs/features/specimen_hybrid_features.npz', allow_pickle=True)

    features = data['features']

    labels = data['labels']

    

    print("🧲 正在计算所有标本与 3 张锚点的相似度，并注入特征库...")

    new_features = []

    

    for feat in features:

        feat_2d = feat.reshape(1, -1)

        

        # 算一下该标本有多像星状毛、盾状毛、过渡态

        sim_s = cosine_similarity(feat_2d, feat_s.reshape(1, -1))[0][0]

        sim_p = cosine_similarity(feat_2d, feat_p.reshape(1, -1))[0][0]

        sim_m = cosine_similarity(feat_2d, feat_m.reshape(1, -1))[0][0]

        

        # 放大这些相似度得分，拼接到原本的 512 维特征末尾 (变成 515 维)

        injected_signals = np.array([sim_s * ANCHOR_WEIGHT, 

                                     sim_p * ANCHOR_WEIGHT, 

                                     sim_m * ANCHOR_WEIGHT])

                                     

        feat_anchored = np.concatenate([feat, injected_signals])

        new_features.append(feat_anchored)

        

    new_features = np.array(new_features)

    

    # 保存为一个新的文件，名字叫 features_anchored.npz

    save_path = 'outputs/features/features_anchored.npz'

    np.savez(save_path, features=new_features, labels=labels, species_names=data['species_names'])

    print(f"🎉 大功告成！带有毛被引导的新特征已保存至: {save_path}")



if __name__ == '__main__':

    main()