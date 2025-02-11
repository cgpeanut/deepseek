import torch
from torch import nn
import json


# Define generator and discriminator architectures
class Generator(nn.Module):
    def __init__(self, input_size=100, output_size=1):
        super().__init__() 
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(512, output_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input) + self.output(input)

class Discriminator(nn.Module):
    def __init__(self, input_size=1):
        super().__init__() 
        self.main = nn.Sequential(
            nn.Conv2d(input_size, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input) + self.output(input)

#def load_and_predict(filepath):
filepath="/home/roxasrr/code/deepseek/powerball_numbers.json"
def load_and_predict(filepath):
    # Load data from JSON file
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    
    # Convert the list of numbers to a torch tensor and reshape it into (n_samples, 1)
    dataset = torch.tensor(dataset).float().view(-1, 1)
    
    # Normalize the dataset - this is just division by the maximum value for simplicity's sake
    max_val = torch.max(torch.abs(dataset))
    dataset /= max_val
    
    # Split into train and test sets
    train_size = int(len(dataset) * 0.67)
    train, test = dataset[:train_size], dataset[train_size:]
    
    # Define the models
    generator = Generator()
    discriminator = Discriminator()
    
    criterion = nn.MSELoss()
    optimizerG = torch.optim.Adam(generator.parameters())
    optimizerD = torch.optim.Adam(discriminator.parameters())
    
    # Train the models
    for epoch in range(20):   # loop over the dataset multiple times
        output = generator(train)
        loss_gen = criterion(output, train)
        
        optimizerG.zero_grad()
        loss_gen.backward()
        optimizerG.step()
        
        real = discriminator(train)
        fake = discriminator(generator(torch.randn(512, 1, 1)))
        loss_discrim = -torch.mean(torch.log(real) + torch.log(1-fake))
        
        optimizerD.zero_grad()
        loss_discrim.backward()
        optimizerD.step()
    
    # Make predictions
    with torch.no_grad():
        #fake = generator(torch.randn(512, 1, 1)).detach().cpu().numpy() * max_val
        fake = generator(torch.randn(512, 1, 1)).detach().gpu().numpy() * max_val
    return fake
