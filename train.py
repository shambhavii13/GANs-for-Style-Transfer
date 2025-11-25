import os
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import argparse
from torchvision import models


# define the dataset class
# This class handles loading images from two directories, A and B, for unaligned datasets.
class UnalignedDataset(Dataset):
    def __init__(self, root, phase):
        self.dir_A = os.path.join(root, phase + 'A')
        self.dir_B = os.path.join(root, phase + 'B')

        self.A_paths = sorted(os.listdir(self.dir_A))
        self.B_paths = sorted(os.listdir(self.dir_B))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

    def __getitem__(self, index):
        A_img = Image.open(os.path.join(self.dir_A, self.A_paths[index % len(self.A_paths)])).convert('RGB')
        B_img = Image.open(os.path.join(self.dir_B, self.B_paths[index % len(self.B_paths)])).convert('RGB')

        return {
            'A': self.transform(A_img),
            'B': self.transform(B_img)
        }


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

# Discriminator definition
# This is a PatchGAN discriminator that classifies 70x70 patches of the input image
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_c=3, ndf=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        ]

        mult = 1
        for i in range(3):
            layers += [
                nn.Conv2d(ndf * mult, ndf * mult * 2, 4, stride=2 if i < 2 else 1, padding=1),
                nn.InstanceNorm2d(ndf * mult * 2),
                nn.LeakyReLU(0.2)
            ]
            mult *= 2

        layers += [nn.Conv2d(ndf * mult, 1, 4, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Training function
def train(args):
    # Setup device and paths
    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    root = "/data/sunny/sham/pytorch-CycleGAN-and-pix2pix/datasets/apple2orange"
    name = "apple2orange"
    dataset = UnalignedDataset(root, "train")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # networks
    netG_A2B = ResnetGenerator().to(device)
    netG_B2A = ResnetGenerator().to(device)
    netD_A = PatchGANDiscriminator().to(device)
    netD_B = PatchGANDiscriminator().to(device)

    # losses
    criterionGAN = nn.MSELoss()
    criterionCycle = nn.L1Loss()
    criterionIdt = nn.L1Loss()

    # optimizers
    opt_G = torch.optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=0.0002, betas=(0.5, 0.999)
    )

    real_label = 1.0
    fake_label = 0.0

    epochs = 200
    for epoch in range(1, epochs+1):
        for i, batch in enumerate(loader):

            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # generator
            opt_G.zero_grad()

            # identity loss
            idt_A = netG_B2A(real_A)
            idt_B = netG_A2B(real_B)
            loss_idt = \
                criterionIdt(idt_A, real_A) * 5.0 + \
                criterionIdt(idt_B, real_B) * 5.0

            # forward cycle
            fake_B = netG_A2B(real_A)
            rec_A = netG_B2A(fake_B)

            fake_A = netG_B2A(real_B)
            rec_B = netG_A2B(fake_A)

            # GAN loss
            loss_G_A2B = criterionGAN(netD_B(fake_B), torch.ones_like(netD_B(fake_B)))
            loss_G_B2A = criterionGAN(netD_A(fake_A), torch.ones_like(netD_A(fake_A)))

            
            # Perceptual loss 
            vgg = models.vgg19(pretrained=True).features.to(device).eval()
            perceptual_loss = nn.MSELoss()
            features_real_A = vgg(real_A)
            features_fake_B = vgg(fake_B)
            loss_perceptual_A2B = perceptual_loss(features_fake_B, features_real_A)
            # Similarly for B2A 
            features_real_B = vgg(real_B)
            features_fake_A = vgg(fake_A)
            loss_perceptual_B2A = perceptual_loss(features_fake_A, features_real_B)  

            # cycle loss
            loss_cycle = criterionCycle(rec_A, real_A) * 10.0 + \
                         criterionCycle(rec_B, real_B) * 10.0

            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle + loss_idt + loss_perceptual_A2B*0.2 + loss_perceptual_B2A*0.2
            loss_G.backward()
            opt_G.step()

            # discriminator
            opt_D.zero_grad()

            # D_A
            loss_D_A_real = criterionGAN(netD_A(real_A), torch.ones_like(netD_A(real_A)))
            loss_D_A_fake = criterionGAN(netD_A(fake_A.detach()), torch.zeros_like(netD_A(fake_A)))
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            # D_B
            loss_D_B_real = criterionGAN(netD_B(real_B), torch.ones_like(netD_B(real_B)))
            loss_D_B_fake = criterionGAN(netD_B(fake_B.detach()), torch.zeros_like(netD_B(fake_B)))
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            opt_D.step()

            print(f"[Epoch {epoch}/{epochs}] [Iter {i}] "
                  f"G: {loss_G.item():.3f} D: {loss_D.item():.3f}")

        # Save
        os.makedirs(f"checkpoints/{args.name}", exist_ok=True)
        torch.save(netG_A2B.state_dict(), f"checkpoints/{args.name}/netG_A2B_{epoch}.pth")
        torch.save(netG_B2A.state_dict(), f"checkpoints/{args.name}/netG_B2A_{epoch}.pth")


# argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/sunny/sham/pytorch-CycleGAN-and-pix2pix/datasets/apple2orange", help="dataset root")
    parser.add_argument("--name", type=str, default="apple2orange", help="name for checkpoints")
    args = parser.parse_args()
    train(args)
