from setuptools import setup, find_packages

setup(
    name='tf_crf_layer',
    version='0.2',
    packages=find_packages(include=['tf_crf_layer', 'tf_crf_layer.*']),
    url='URL',
    license='MIT',
    install_requires=[
          'tensorflow',
    ],
    tests_require=['pytest', 'numpy', 'keras'],
    author='Xiaoquan Kong',
    author_email='u1mail2me@gmail.com',
    description='CRF layer for TensorFlow 1.x'
)
