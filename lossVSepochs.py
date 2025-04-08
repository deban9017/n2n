import matplotlib.pyplot as plt

# Extracted from the logs
# Forgot to save the logs, so I had to extract them from the console output.
# logs under training cell in n2n.ipynb
train_loss = [
    0.0880, 0.0173, 0.0151, 0.0145, 0.0138, 0.0136, 0.0127, 0.0125, 0.0123, 0.0122,
    0.0122, 0.0120, 0.0120, 0.0115, 0.0116, 0.0115, 0.0116, 0.0117, 0.0113, 0.0113,
    0.0111, 0.0112, 0.0110, 0.0111, 0.0110, 0.0110, 0.0109, 0.0108, 0.0107, 0.0110,
    0.0110, 0.0108, 0.0106, 0.0106, 0.0107, 0.0105, 0.0104, 0.0108, 0.0105, 0.0105,
    0.0105, 0.0106, 0.0104, 0.0105, 0.0103, 0.0102, 0.0103, 0.0104, 0.0106, 0.0103
]

val_loss = [
    0.0085, 0.0044, 0.0044, 0.0033, 0.0045, 0.0030, 0.0027, 0.0027, 0.0025, 0.0026,
    0.0025, 0.0023, 0.0027, 0.0022, 0.0026, 0.0021, 0.0021, 0.0030, 0.0021, 0.0023,
    0.0022, 0.0019, 0.0020, 0.0019, 0.0020, 0.0019, 0.0021, 0.0021, 0.0023, 0.0021,
    0.0019, 0.0021, 0.0017, 0.0019, 0.0020, 0.0019, 0.0019, 0.0022, 0.0018, 0.0018,
    0.0019, 0.0020, 0.0017, 0.0019, 0.0018, 0.0025, 0.0017, 0.0017, 0.0019, 0.0016
]

epochs = list(range(1, 51))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train & Validation Loss vs Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_vs_epochs.png')
plt.show()
