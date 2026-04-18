import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_2 = nn.LazyLinear(128)
        self.d_5 = nn.Dropout(0.3)
        self.l_6 = nn.LazyLinear(10)

    def forward(self, x):
        v_0 = x
        v_2 = self.l_2(v_0)
        v_3 = torch.relu(v_2)
        v_5 = self.d_5(v_3)
        v_6 = self.l_6(v_5)
        v_7 = torch.softmax(v_6, dim=1)
        return v_7


if __name__ == '__main__':
    print("Initializing Classifier...")
    model = Classifier()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Generating simulated dataloader...")
    input_shape = 784
    X = torch.randn(100 * 32, input_shape)
    y = torch.randint(0, 10, (100 * 32,))

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

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
