import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_size=100, output_size=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_size),
            nn.Tanh()  # 将输出限制在[-1, 1]之间，适合生成图像数据
        )

    def forward(self, x):
        return self.model(x)


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_size=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出一个介于0到1之间的概率值
        )

    def forward(self, x):
        return self.model(x)


# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=64, shuffle=True)

# 初始化模型
generator = Generator()
discriminator = Discriminator()

# 损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# 训练过程
for epoch in range(50):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)

        # 训练判别器
        real_images = real_images.view(batch_size, -1)  # 展平图像
        real_labels = torch.ones(batch_size, 1)  # 真实标签为1
        fake_labels = torch.zeros(batch_size, 1)  # 假标签为0

        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)  # 判别器对真实图像的损失

        noise = torch.randn(batch_size, 100)  # 随机噪声
        fake_images = generator(noise)  # 生成假图像
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)  # 判别器对假图像的损失

        d_loss = d_loss_real + d_loss_fake  # 总损失
        optimizer_d.zero_grad()
        d_loss.backward()  # 反向传播
        optimizer_d.step()  # 更新判别器

        # 训练生成器
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # 生成器希望判别器输出1

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(f'Epoch [{epoch + 1}/50], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
