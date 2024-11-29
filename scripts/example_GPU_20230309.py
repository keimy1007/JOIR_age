"""
@author: Takuro Kamei (Kyoto Univ.)

This is a sample code for estimating retinal age from fundus images using deep learning. 

2023/3/9 推論時に全く同じ画像を入力しても実行ごとに結果が少し変わってしまうという問題点を解決しました。
"""

# 必要ライブラリ読み込み
import torch
import timm
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm

# モデル枠組み読み込み
model = timm.create_model(model_name='swin_base_patch4_window12_384', num_classes=1, pretrained=False)

# GPU使用する場合
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 学習済みモデル読み込み
model_path = 'model_20220903.pth'
model.load_state_dict(torch.load(model_path))

# 乱数固定化の定義
def torch_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    
# 画像データのパス(list)を指定
img_list = ['test1.jpg', 'test2.jpg']

# imageNetに合わせた画像の正規化
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# transformの定義
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

# Datasetクラスの作成
class Dataset(data.Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        img = self.transform(img)   
        return img

# Datasetの作成
dataset = Dataset(
    img_list=img_list, transform=transform
)

# Dataloaderの作成
# batch sizeはGPU(CPU)性能に応じて適宜変更して下さい
loader = data.DataLoader(
    dataset, batch_size=1, shuffle=False
)

# 乱数固定化
torch_seed()

# 年齢予測
pred_r = []

model.eval()
with torch.no_grad():
    for inputs in tqdm(loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        pred_r.append(outputs.data.cpu().numpy())
        
pred = np.concatenate(pred_r)

# 結果出力
print(pred)