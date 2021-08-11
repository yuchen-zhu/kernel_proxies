import os,sys
filepath = os.path.realpath(__file__)
directory, filename = os.path.split(filepath)
pmmr_root_path = os.path.dirname(directory)
sys.path.append(pmmr_root_path)  # append pmmr root path

project_root_path = os.path.join(pmmr_root_path, '..')
sys.path.append(project_root_path)  # append systems root path

