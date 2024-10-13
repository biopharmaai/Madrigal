from setuptools import setup

setup(name='madrigal',  # change this back to `novelddi` if you are using original notebooks/scripts
      version='0.1',
      description='Multimodal learning for drug combination outcome prediction',
      packages=['novelddi'],
      author = 'zitnik-lab',
      zip_safe=False, 
      python_requires='>=3.8')