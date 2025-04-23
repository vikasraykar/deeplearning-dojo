"""Logistic Regression using pytorch."""

# %% [markdown]
# **Logistic Regression using pytorch.**

# %%
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def describe_tensor(X, name):
    print(f"{name} - shape - {X.shape} - dtype - {X.dtype} - device - {X.device}")
    # print(X)


# %%
# Sample train and test data.
def sample_data(N: int = 1000, d: int = 2, train_size: float = 0.7):
    """Generate sample train and test data for binary classification.

    Args:
        N (int, optional): Number of samples. Defaults to 1000.
        d (int, optional): Number of features. Defaults to 2.
        train_size (float, optional): Fraction of instances in training. Defaults to 0.7.

    Returns:
        X_train : Features matrix for the train split. (N_train,d)
        y_train : Target vector for the train split. (N_train,)
        X_test : Features matrix for the test split. (N_test,d)
        y_test : Target vector for the test split. (N_test,)
    """
    N_positive = int(N / 2)
    N_negative = N - N_positive

    X_positive = np.random.multivariate_normal(
        mean=np.ones(d), cov=np.eye(d), size=N_positive
    )
    y_positive = np.ones(N_positive)

    X_negative = np.random.multivariate_normal(
        mean=-np.ones(d), cov=np.eye(d), size=N_negative
    )
    y_negative = np.zeros(N_negative)

    X = np.concatenate([X_positive, X_negative])
    y = np.concatenate([y_positive, y_negative])

    # Split into train and test.
    indices = np.random.permutation(N)
    N_train = int(N * train_size)
    train_idx, test_idx = indices[:N_train], indices[N_train:]
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, y_train, X_test, y_test


N = 10000
d = 3
X_train, y_train, X_test, y_test = sample_data(N, d)

print(X_train.shape)
print(y_train.shape)


# %%
# Create a Custom Dataset class.
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
class SampleDataset(Dataset):
    """A dataset must implement the following 3 functions."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Initialization."""
        self.N, self.d = X.shape
        # Convert from numpy array to torch tensors.
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(self.N, 1)
        describe_tensor(self.X, "X")
        describe_tensor(self.y, "y")

    def __len__(self):
        """Returns the number of samples in our dataset."""
        return self.N

    def __getitem__(self, idx: int):
        """Returns a sample from the dataset at the given index idx."""
        return self.X[idx, :], self.y[idx, :]


train_dataset = SampleDataset(X_train, y_train)
test_dataset = SampleDataset(X_test, y_test)
print(len(train_dataset))
print(train_dataset[0])

# %%
# Preparing your data for training with DataLoaders.
# The Dataset retrieves our dataset’s features and labels one sample at a time.
# While training a model, we typically want to pass samples in mini-batches,
# reshuffle the data at every epoch to reduce model overfitting, and use Python’s
# multiprocessing to speed up data retrieval.m
# DataLoader is an iterable that abstracts this complexity for us in an easy API.
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader to access mini-batches.
# Once we have loaded that dataset into the DataLoader we can iterate through the dataset
# as needed. Each iteration below returns a batch of train_features and train_labels
# (containing batch_size=64 features and labels respectively).
# Because we specified shuffle=True, after we iterate over all batches the data is shuffled.
for X, y in train_dataloader:
    describe_tensor(X, name="X")
    describe_tensor(y, name="y")
    break

# %%
# Get device/accelerator for training.
# We want to be able to train our model on an accelerator such as CUDA, MPS etc.
# If the current accelerator is available, we will use it. Otherwise, we use the CPU.
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} accelerator.")


# %%
# Define the model class.
# We define our neural network by subclassing nn.Module, and initialize the neural network
# layers in __init__. Every nn.Module subclass implements the operations on input data in
# the forward method.
class LogisticRegression(nn.Module):
    """A model class should implement the init and forward functions."""

    def __init__(self, num_features: int):
        """Logistic Regression.

        Args:
            num_features (int): Number of features.
        """
        super().__init__()
        self.linear = nn.Linear(in_features=num_features, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_predicted = self.sigmoid(self.linear(x))
        return y_predicted


model = LogisticRegression(num_features=d)
model.to(device)
print(f"Model structure: \n{model}\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}")

# %%
# Loss function.
# Binary Cross Entropy Loss.
loss_fn = nn.BCELoss()

# %%
# Optimizer.
# SGD with momentum.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-08)

# %%
# Learning rate scheduler.
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


# %%
# Testing
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += ((pred >= 0.5) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
    print(f"Test Error: Accuracy: {(100.0*correct):>0.1f} Avg loss: {test_loss:>5f} \n")


# %%
# Training.
num_epochs = 10
num_batches = len(train_dataloader)
for epoch in range(0, num_epochs):
    test(test_dataloader, model, loss_fn)

    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Forward propagation.
        y_pred = model(X)

        # Loss function.
        loss = loss_fn(y_pred, y)

        # Back propagation to compute gradient.
        loss.backward()

        # Gradient descent.
        optimizer.step()

        # Prevent gradient accumulation
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            print(
                f"epoch {epoch:>3d}/{num_epochs:>3d} batch {batch:>3d}/{num_batches:>3d} batch loss: {loss:>5f} "
            )

    # Adjust learning rate
    print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
    scheduler.step()

# %%
# Save the model.
torch.save(model.state_dict(), "model.pth")

# Load the model for testing.
model = LogisticRegression(num_features=d).to(device)
model.load_state_dict(torch.load("model.pth", weights_only="True"))

# %%
