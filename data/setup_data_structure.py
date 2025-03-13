import os
import sys

def create_data_directories():
    """
    Create the necessary data directories for the project.
    """
    # Define the base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    
    # Create data subdirectories
    directories = [
        os.path.join(base_dir, 'raw'),
        os.path.join(base_dir, 'interim'),
        os.path.join(base_dir, 'processed')
    ]
    
    # Create each directory if it doesn't exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    
    # Create a .gitkeep file in each directory to maintain structure in git
    for directory in directories:
        gitkeep_file = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_file):
            with open(gitkeep_file, 'w') as f:
                pass
            print(f"Created .gitkeep in {directory}")

if __name__ == "__main__":
    create_data_directories()
    print("Data directory structure setup complete.")