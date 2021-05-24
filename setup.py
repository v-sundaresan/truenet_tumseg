from setuptools import setup, find_packages
setup(python_requires='>=3.6')
with open('requirements.txt', 'rt') as f:
    install_requires = [l.strip() for l in f.readlines()]

setup(name='truenet_tumseg',
	  version='1.0.1',
	  description='DL method for brain tumour segmentation',
	  author='Vaanathi Sundaresan',
	  install_requires=install_requires,
	  scripts=['truenet_tumseg/scripts/truenet_tumseg', 'truenet_tumseg/scripts/prepare_tumseg_data'],
	  packages=find_packages(),
	  include_package_data=True)
