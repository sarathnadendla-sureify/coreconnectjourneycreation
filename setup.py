from setuptools import find_packages,setup

setup(name="journeys",
       version="0.0.1",
       author="sarath",
       author_email="sarathnadendla88@gmail.com",
       packages=find_packages(),
       install_requires=['lancedb','langchain','langgraph','tavily-python','polygon']
       )