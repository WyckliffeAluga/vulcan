
from jetbot import Camera
import torchvision.transforms as transforms
from dataset import ImageClassificationDataset

camera = Camera.instance()

TASK = 'thumbs'
#task = 'emotions'
#task = 'fingers'
#task = 'diy'

CATEGORIES = ['thumbs_up', 'thumbs_down']
# categories = ['none', 'happy', 'sad', 'angry']
# categories = ['1', '2', '3', '4', '5']
# categories = [ 'diy_1', 'diy_2', 'diy_3']

DATASETS = ['A', 'B']
#datasets =['A', 'B', 'C']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
datasets = {}
for name in DATASETS:
    datasets[name] = ImageClassificationDataset(TASK + '_' + name, CATEGORIES, TRANSFORMS)

print("{} task with {} categories defined".format(TASK, CATEGORIES))
