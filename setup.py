from setuptools import setup, find_packages

setup(
    name="julia",
    version="0.1",
    author="Kaito Arai, Rowan Hobson, Harrison Mouat, Jan Panek, Baptiste Simandoux",
    author_email="kaito.arai19@imperial.ac.uk, rowan.hobson19@imperial.ac.uk, harrison.mouat19@imperial.ac.uk, jan.panek19@imperial.ac.uk, baptiste.simandoux19@imperial.ac.uk",
    description="A package for drawing Julia sets and external/internal rays associated with cubic maps and the Newton maps associated with them.",
    url="https://github.com/Harry-icl/M2R-Julia-Sets",
    project_urls={
        "Bug Tracker": "https://github.com/Harry-icl/M2R-Julia-Sets/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'numpy',
        'matplotlib',
        'numba',
        'Pillow',
        'opencv-python',
        'PySimpleGUI',
    ],
)
