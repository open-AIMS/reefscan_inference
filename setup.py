from setuptools import setup
setup(
    name='inferencer',
    packages=['inferencer', 'inferencer.utils', 'inferencer.utils.old', 'inferencer.models', 'inferencer.models.classifier', 'inferencer.models.ft_ext'],
    package_data={
        'inferencer.models.ft_ext': ['weights.best.hdf5'],
        'inferencer.models.classifier': ['reefscan.sav'],
        'inferencer.models': ['reefscan_group_labels.csv'],
    },
    version='1.1.4',
    description='aims_reefscan_inferencer',
    author='AIMS',
    license='MIT',
)