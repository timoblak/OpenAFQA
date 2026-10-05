from setuptools import setup, find_packages

setup(
    name='afqa_toolbox',
    version='0.1.0',
    url='https://github.com/timoblak/OpenAFQA',
    author='Tim Oblak',
    author_email='tim.oblak@fri.uni-lj.si',
    description='An Automated Fingermark Quality Assessment Toolbox',
    packages=find_packages(include=['afqa_toolbox', 'afqa_toolbox.*']),
    package_data={'afqa_toolbox': ['resources/*.pkl', 'resources/*.pt']},
    python_requires='>=3.8',
    install_requires=['numpy >= 1.21',
                      'scipy >= 1.7',
                      'scikit-image >= 0.19',
                      'scikit-learn >= 0.24',
                      'matplotlib >= 3.5',
                      'opencv_python >= 4.5',
                      'torch >= 1.13',
                      'torchvision >= 0.14'],
)
