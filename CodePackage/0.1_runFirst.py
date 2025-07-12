## RUN SETUP
# This script sets up the environment for the TransientBloodRheo project.
# It installs necessary packages and configures the environment.

import os
import sys
import subprocess
import platform

def setup_virtual_environment(venv_name="my_env", force_recreate=False):
    """
    Sets up a virtual environment for the project.
    
    Parameters:
    venv_name (str): Name of the virtual environment.
    force_recreate (bool): If True, recreate the virtual environment if it exists.
    """
    venv_path = os.path.join(os.getcwd(), venv_name)
    
    if force_recreate and os.path.exists(venv_path):
        subprocess.run([sys.executable, '-m', 'venv', '--clear', venv_path])
    elif not os.path.exists(venv_path):
        subprocess.run([sys.executable, '-m', 'venv', venv_path])
    
    # Activate the virtual environment
    activate_script = os.path.join(venv_path, 'Scripts', 'activate') if platform.system() == 'Windows' else os.path.join(venv_path, 'bin', 'activate')
    
    print(f"Virtual environment '{venv_name}' is set up at {venv_path}. Activate it using: source {activate_script}")

    # Install required packages
    requirements_file = os.path.join(os.getcwd(), 'requirements.txt')
    if os.path.exists(requirements_file):
        subprocess.run([os.path.join(venv_path, 'bin', 'pip'), 'install', '-r', requirements_file])
        print(f"Installed packages from {requirements_file}.")
    else:
        print(f"Requirements file {requirements_file} not found. No packages installed.")
    return venv_path

if __name__ == "__main__":
    setup_virtual_environment(venv_name="my_env", force_recreate=False)