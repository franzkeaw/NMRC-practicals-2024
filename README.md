This repository contains the computer practicals for the course
Neural Models, Representation and Consciousness, as part of the 
Master Biomedical Sciences, track Cognitive Neurobiology and Clinical Neurophysiology, 
of the University of Amsterdam. 

# Requirements
In this practical, we will rely on using python, which is a popular programming language that makes it easy to write and understand code.
Furthermore, we will use Jupyter Notebooks, which is a special tool that lets you write and run Python code in small, easy-to-read sections in your web browser. It’s like a digital notebook where you can mix text, (python-)code, and results all in one place. We have created one notebook for each practical, which you will download from this Gitlab page (see instructions below)

That means that you need to install Python and Jupyter Notebooks, and there are many ways to do so. If you are new to most of this, we recommend to install the [Anaconda Distribution](https://www.anaconda.com/download/) as this will make a lot of things easier for you.
Anaconda is a program that helps you easily manage Python and all the tools you might need for coding. It’s like a toolbox that comes with everything set up, so you don’t have to worry about installing things one by one. Anaconda includes Python, Jupyter Notebooks, and lots of useful packages for tasks like data science, machine learning, or scientific computing.
Note that the website asks you to register with your email, but you can also click on "skip registration". 
 
Once you have Anaconda installed, you need to create a python environment for this practical. A python environment is like a workspace where you can write and run Python code. Imagine it as a special box where you keep everything needed to work on a project. Inside the box, there’s the Python program itself, plus any extra tools (called packages or libraries) that help you with specific tasks like math, data analysis, or making websites. When you create an environment, a folder will be created on your computer, in which we will place all the code and files which we need to run the practical. We have everything prepared for you and packed everything into a file, so you dont need to worry about any of this if you folloe the instructions below:

# Instructions
- Download Anaconda from the link given above, and be sure to select the Python 3.7 version.
- On the Gitlab page ( https://gitlab.com/csnlab/nmrc-practicals/-/tree/updates_2024 ), click on: Code -> Download source code -> zip, and extract the zip file to a location on your laptop.
- Open Anaconda
- Go to the 'Environments' section
- Click 'Import'
- Select the path to the `nmrc2024.yml` file, which is in the zip file which you just downloaded, and choose a name for you environment (default will be `nmrc2024`).

Once your environment has been created:
- In Anaconda on the left you find "environments" -> Click on the environment name which you just created (a green triangle should appear)
- Click on the green triangle
- Select 'Open with Jupyter notebook'. This will start a Jupyter notebook in your home directory. 
- Navigate to the folder of the practicals. 
    Note: If you are unable to navigate to your folder, you can also choose another option to open the jupyter notebooks:
    - Instead of....

In case you get 

```Widget Javascript not detected```

You may need to enable the widget extension with 

```$ jupyter nbextension enable --py --sys-prefix widgetsnbextension```

```$ jupyter nbextension enable --py widgetsnbextension```
