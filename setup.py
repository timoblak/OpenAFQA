from setuptools import setup, find_packages

setup(
    name='afqa_toolbox',
    version='0.0.1',
    url='https://github.com/timoblak/OpenAFQA',
    author='Tim Oblak',
    author_email='tim.oblak@fri.uni-lj.si',
    description='An Automated Fingermark Quality Assessment Toolbox',
    packages=find_packages(),    
    install_requires=['numpy >= 1.18',
                      'scipy >= 1.7.0',
                      'scikit-image >= 0.18',
                      'scikit-learn >= 0.24',
                      'matplotlib >= 3.4',
                      'opencv_python >= 4.5'],
)


