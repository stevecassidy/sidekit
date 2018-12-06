from distutils.core import setup

# patch distutils if it can't cope with the "classifiers" or
# "download_url" keywords
from sys import version
if version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None

setup(
    name='SIDEKIT',
    version='1.2.6',
    author='Anthony Larcher',
    author_email='anthony.larcher@univ-lemans.fr',
    packages=['sidekit', 'sidekit.bosaris', 'sidekit.frontend', 'sidekit.libsvm', 'sidekit.nnet'],
    url='http://www-lium.univ-lemans.fr/sidekit/',
    download_url='http://pypi.python.org/pypi/Sidekit/',
    license='LGPL',
    platforms=['Linux, Windows', 'MacOS'],
    description='Speaker, Language Recognition and Diarization package.',
    long_description=open('README.txt').read(),
    install_requires=[
        "mock>=1.0.1",
        "nose>=1.3.4",
        "numpy>=1.11",
        "pyparsing >= 2.0.2",
        "python-dateutil >= 2.2",
        "scipy>=0.12.1",
        "six>=1.8.0",
        "matplotlib>=1.3.1",
        "pytorch >= 0.4",
	"PyYAML>=3.11",
	"h5py>=2.5.0",
	"pandas>=0.16.2"
    ],
    package_data={'sidekit': ['docs/*']},
    classifiers=['Development Status :: 4 - Beta',
                 'Environment :: Console',
                 'Environment :: MacOS X',
                 'Environment :: Win32 (MS Windows)',
                 'Environment :: X11 Applications',
                 'Intended Audience :: Education',
                 'Intended Audience :: Legal Industry',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
                 'Natural Language :: English',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft',
                 'Operating System :: Other OS',
                 'Operating System :: POSIX',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Multimedia :: Sound/Audio :: Speech',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence']
)




