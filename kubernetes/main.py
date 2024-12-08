import pandas as pd
import random
import os


class MLProjectSetup:
    def __init__(self, base_path, structure):
        self.base_path = base_path
        self.structure = structure

    def validate_and_create_structure(self):
        """
        Validate the directory structure against the expected structure.
        Create any missing directories or files.
        """
        for folder, contents in self.structure.items():
            folder_path = os.path.join(self.base_path, folder)

            # Ensure the folder exists and is a directory
            if os.path.exists(folder_path):
                if not os.path.isdir(folder_path):
                    print(f"Path exists but is not a directory. Removing: {folder_path}")
                    os.remove(folder_path)  # Remove the file if it conflicts with the directory
                    os.makedirs(folder_path)
                    print(f"Created directory: {folder_path}")
                else:
                    print(f"Directory already exists: {folder_path}")
            else:
                os.makedirs(folder_path)
                print(f"Created missing directory: {folder_path}")

            # Validate and create contents (files and subdirectories)
            for content in contents:
                try:
                    file_path = os.path.join(folder_path, content)

                    # Check if it's a file
                    if "." in content:  # A simple heuristic to detect files (e.g., ".py", ".yaml")
                        if not os.path.exists(file_path):
                            open(file_path, 'w').close()
                            print(f"Created missing file: {file_path}")
                    else:  # It's a subdirectory
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)
                            print(f"Created missing subdirectory: {file_path}")

                except Exception as e:
                    print(f"Error while processing {content}: {e}")

        print(f"Directory structure validated and ensured under {self.base_path}")

    def generate_dataset(self, n, output_file="dataset.csv"):
        """
        Generate a dataset and save it to the 'data' folder.
        """
        # Ensure the data folder exists and is a directory
        data_folder = os.path.join(self.base_path, "k8s-ml-pipeline", "data")

        if os.path.exists(data_folder):
            if not os.path.isdir(data_folder):
                print(f"Path exists but is not a directory. Removing: {data_folder}")
                os.remove(data_folder)  # Remove the file if it conflicts with the directory
                os.makedirs(data_folder)
                print(f"Created directory: {data_folder}")
            else:
                print(f"Data folder already exists: {data_folder}")
        else:
            os.makedirs(data_folder)
            print(f"Created missing data folder: {data_folder}")

        # Generate random dataset
        data = []
        for _ in range(n):
            square_footage = random.randint(800, 3000)
            bedrooms = random.randint(1, 5)
            bathrooms = random.randint(1, 3)
            price = square_footage * random.randint(100, 200) + bedrooms * 5000 + bathrooms * 10000
            data.append({
                "square_footage": square_footage,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "price": price
            })

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Define the full path for the output file
        output_path = os.path.join(data_folder, output_file)

        try:
            # Save the dataset as a CSV file
            df.to_csv(output_path, index=False)
            print(f"Dataset with {n} rows saved to {output_path}.")
        except Exception as e:
            print(f"Error while saving dataset: {e}")


# Define the directory structure
structure = {
    "k8s-ml-pipeline": {
        "Dockerfiles": ["Dockerfile.train", "Dockerfile.predict"],
        "scripts": ["train.py", "predict.py"],
        "data": ["dataset.csv"],  # Placeholder
        "yaml": ["train-job.yaml", "predict-deployment.yaml"]
    }
}

# Set up and ensure files and folders exist
base_path = os.getcwd()  # Current working directory
project_setup = MLProjectSetup(base_path, structure)

# Validate and create the directory structure
project_setup.validate_and_create_structure()

# Generate the dataset
project_setup.generate_dataset(100)
