import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

data_frame = pd.read_csv("loto_649_si_noroc.csv")
# so far vom merge pe BCE pentru ca pot fi mai multe variante simultan
print(data_frame.head())
#feature engineering, scapam de coloana date pentru ca nu ne trebe
data_frame_final = data_frame.drop(columns=['date'])
print(data_frame_final.head())



def encode_single_draw(draw_numbers):
    encoded_vector = [0] * 49

    for number in draw_numbers:
        index = number - 1
        encoded_vector[index] = 1 #sa avem pozitia corecta
    return encoded_vector

first_extraction = data_frame_final.iloc[0].tolist()
print("first extraction: ", first_extraction)

encoded_vector = encode_single_draw(first_extraction)
print("encoded vector: ", encoded_vector)

def encode_all_draws(df_values):
    all_encoded_vector = []

    for row in df_values:
        encoded_row = encode_single_draw(row)
        all_encoded_vector.append(encoded_row)
    return np.array(all_encoded_vector)

extracted_values = data_frame_final.values
dataset = encode_all_draws(extracted_values)


print("extracted values(before training ofc): ", dataset[:2])

target_x = dataset[:-1]
target_y = dataset[1:]

print("shapes: ", target_x.shape, "\n", target_y.shape)

x_tensor = torch.tensor(target_x).float()
y_tensor = torch.tensor(target_y).float()
print("x_tensor: ", x_tensor)
print("y_tensor: ", y_tensor)

class SimpleNet(torch.nn.Module):
    def __init__(self):
        self.fc1 = torch.nn.Linear(3, 3)
        self.fc2 = torch.nn.Linear(3, 2)
        self.fc3 = torch.nn.Linear(2, 1)

class LotoNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(49, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 49)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = torch.sigmoid(self.fc3(x))
        return x

model = LotoNet()
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 250
print("epochs: ", epochs)
print("TRAIN HHHRRRRRRAAAAGHHHHHH! BIG BRAAAIN! ")
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(x_tensor)
    loss = criterion(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    if(epoch + 1) % 10 == 0 or (epoch == 0):
        print("Epoch: ", epoch + 1/epochs)
        print("Loss: ", loss.item())

model.eval()
recent_extraction_raw = extracted_values[-1]
recent_extraction_encoded = dataset[-1]

input_tensor = torch.tensor(recent_extraction_encoded).float()

with torch.no_grad():
    predictions = model(input_tensor)

probabilities = predictions.numpy()
top_6_indexes = np.argsort(probabilities)[-6:][::-1]
predicted_numbers = [int(index) + 1 for index in top_6_indexes]
predicted_numbers_sorted = sorted(predicted_numbers)

print("*drum roll*")
print("*more drum rolls*")

print("last known extraction: ", sorted(recent_extraction_raw.tolist()), " ")
print("\n *even more drum rolls*")
print("predicted extraction...", predicted_numbers_sorted)

