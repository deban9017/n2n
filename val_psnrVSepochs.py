import matplotlib.pyplot as plt

# Validation PSNR values for 50 epochs
# Extracted from the logs
# logs under training cell in n2n.ipynb
val_psnr = [
    21.13, 23.75, 24.01, 25.09, 24.09, 25.62, 25.90, 25.88, 26.30, 26.08,
    26.25, 26.53, 26.04, 26.73, 25.98, 26.91, 26.87, 25.39, 27.03, 26.62,
    26.73, 27.32, 27.15, 27.53, 27.24, 27.37, 27.00, 27.16, 26.84, 27.00,
    27.30, 26.93, 27.81, 27.50, 27.26, 27.46, 27.45, 26.76, 27.64, 27.71,
    27.52, 27.10, 27.83, 27.31, 27.65, 27.06, 27.84, 27.91, 27.31, 28.04
]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), val_psnr, label="Validation PSNR", color='green', marker='o')
plt.title("Validation PSNR vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('val_psnr_vs_epochs.png')
# plt.show()
