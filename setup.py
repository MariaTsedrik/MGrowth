from setuptools import setup, find_packages


#def get_requires():
#    reqs = []
#    for line in open("requirements.txt", "r").readlines():
#        reqs.append(line)
#    return reqs

setup(
    name="MGrowth",
    version=0.1,
    url="https://github.com/MariaTsedrik/MGrowth",
    author="Maria Tsedrik",
    author_email="",
    license="BSD",
    packages=find_packages(),
    keywords="Modified Gravity Extended Cosmology",
#    install_requires=get_requires(),
)