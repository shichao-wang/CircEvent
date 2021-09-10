import setuptools

setuptools.setup(
    name="circumst_event",
    version="1.0.0",
    packages=[setuptools.find_packages("src")],
    package_dir={"": "src"},
    url="https://github.com/Shichao-Wang/CircEvent",
    license="",
    author="Shichao Wang",
    author_email="wangshichao@dbis.nankai.edu.cn",
    description="Incorporating Circumstance into Narrative Event Prediction",
    extra_requires={"dev": []},
)
