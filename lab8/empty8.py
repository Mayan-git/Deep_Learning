from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

X = np.random.rand(100, 20)

inputs = Input(shape=(20,))
encoded = Dense(10, activation='relu')(inputs)
latent = Dense(2)(encoded)

decoded = Dense(10, activation='relu')(latent)
outputs = Dense(20, activation='sigmoid')(decoded)

vae = Model(inputs, outputs)
vae.compile(optimizer='adam', loss='mse')

vae.fit(X, X, epochs=3, batch_size=8)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

generator = Sequential([
    Dense(16, activation='relu', input_dim=10),
    Dense(1, activation='sigmoid')
])

discriminator = Sequential([
    Dense(16, activation='relu', input_dim=1),
    Dense(1, activation='sigmoid')
])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

real_data = np.random.rand(100, 1)
noise = np.random.rand(100, 10)

fake_data = generator.predict(noise)

discriminator.fit(real_data, np.ones((100, 1)), epochs=1)
discriminator.fit(fake_data, np.zeros((100, 1)), epochs=1)


import torch
import torch.nn as nn

class SimpleGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.fc(x)
        return x


x = torch.rand(5, 10)        
adj = torch.eye(5)            

model = SimpleGCN()
output = model(x, adj)

print("GCN Output:\n", output)
