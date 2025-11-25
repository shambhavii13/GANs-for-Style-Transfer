import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=str, default="10", help="epoch number to load")
parser.add_argument("--name", type=str, default="apple2orange", help="name of dataset")
parser.add_argument("--model_name", type=str, default="vangogh2photo_loss", help="name of the model checkpoint")
parser.add_argument("--direction", type=str, default="A2B", help="A2B or B2A")
args = parser.parse_args()

# Resnet block definition
# This is a basic ResNet block used in the generator architecture.
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)
    
# ResnetGenerator definition
class ResnetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3, ngf=64, n_blocks=6):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        # downsample
        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
            mult *= 2

        # res blocks
        for _ in range(n_blocks):
            layers += [ResnetBlock(ngf * mult)]

        # upsample
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True)
            ]
            mult //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_c, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Setup device and paths
device = "cuda:2" if torch.cuda.is_available() else "cpu"
dataset_root = f"/data/sunny/sham/pytorch-CycleGAN-and-pix2pix/datasets/{args.name}" # Path to the dataset # change as needed

output_dir = f"./results/{args.model_name}/{args.direction}/"
os.makedirs(output_dir, exist_ok=True)

# load the generator model
def load_generator(path):
    netG = ResnetGenerator().to(device)
    state = torch.load(path, map_location=device)
    netG.load_state_dict(state)
    netG.eval()
    return netG
# Determine the direction and load the appropriate generator
if args.direction == "A2B":
    netG = load_generator(f"checkpoints/{args.model_name}/netG_A2B_{args.epoch}.pth")
    test_folder = os.path.join(dataset_root, "testA")
else:
    netG = load_generator(f"checkpoints/{args.model_name}/netG_B2A_{args.epoch}.pth")
    test_folder = os.path.join(dataset_root, "testB")

# transform for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def save_image(tensor, path):
    img = (tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1).cpu()
    img = transforms.ToPILImage()(img)
    img.save(path)

# Process each image in the test folder
for fname in os.listdir(test_folder):
    if fname.lower().endswith(("jpg", "png")):
        inp_path = os.path.join(test_folder, fname)
        out_path = os.path.join(output_dir, f"fake_{fname}")

        img = load_image(inp_path)
        with torch.no_grad():
            fake = netG(img)
        save_image(fake, out_path)
        print(f"Processed {fname}")

print("Inference completed!")

