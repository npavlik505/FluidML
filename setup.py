from setuptools import setup, find_packages

# #Address when a README.md file is created
# with open(file = 'README.md', mode = 'r') as readme_handle:
#     long_description = readme_handle.read()

setup(
    name = 'FluidML',
    author = 'Nate Pavlik',
    author_email = 'nathanielpavlik@gmail.com'
    #If version number listed elsewhere, explore linking the two to guarentee both are updated together
    #Read version as:
    #   -Major Version 0
    #   -Minor Version 1
    #   -Maintenance Version 0
    version = '0.1.0',

    description = 'A python platform for testing control, modelling, and feature selection machine learning algorithms on a library of fluid flows'
    # Uncomment when README.md file is created
    #long_description = long_description,
    long_description_content_type = 'text/markdown'
    url = 'https://github.com/npavlik505/FluidML'
    install_requires = [
        'numpy == '
        'matplotlib == '
        'pyplot == '
        'seaborn == '
        'ptitprince == '
        'torch == '
        'h5py == '
        'pysindy == '
        'gym == '
    ],
    keywords = 'fluid dynamics', 'machine learning', 'reinforcement learning',
    # Replace 'folder_containing_install_packages' with name of foler (to be created) containing all the packages that need to be installed
    packages = find_packages(
        where = ['folder_containing_install_packages', 'folder_containing_install_packages.*']
    ),

    python_requires = '>= '

    classifiers = [
        'Development Status :: ',
        'Intended Audience :: Research',
        'License :: :: '
        'Natural Language :: English'
        'Operating System :: OS Independent'
        'Programming Language :: Python'
        'Topic :: Fluid Dynamics', 
        'Topic :: Machine Learning' 

    ]
)

#REFERENCED WHEN CREATING setup.py FILE
#https://www.youtube.com/watch?v=-hENwoex93g