import kagglehub

# Download latest version
path = kagglehub.dataset_download("gagandeep16/car-sales")

print("Path to dataset files:", path)