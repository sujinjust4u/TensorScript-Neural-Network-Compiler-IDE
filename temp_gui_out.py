import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_2 = nn.LazyLinear(128)
        self.d_4 = nn.Dropout(0.3)
        self.l_5 = nn.LazyLinear(10)

    def forward(self, x):
        v_0 = x
        v_2 = self.l_2(v_0)
        v_3 = torch.relu(v_2)
        v_4 = self.d_4(v_3)
        v_5 = self.l_5(v_4)
        v_6 = torch.softmax(v_5, dim=1)
        return v_6


if __name__ == '__main__':
    print("Initializing Classifier...")
    model = Classifier()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Generating simulated dataloader...")
    input_shape = 784
    X = torch.randn(100 * 30, input_shape)
    y = torch.randint(0, 10, (100 * 30,))

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=30)

    print("Starting Training Loop...")
    for epoch in range(5):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/5], Loss: {epoch_loss/len(dataloader):.4f}")
    print("Training Complete!")
