This repository contains the computer practicals for the course
Neural Models, Representation and Consciousness, as part of the 
Master Biomedical Sciences, track Cognitive Neurobiology and Clinical Neurophysiology, 
of the University of Amsterdam.

A special thanks to Pietro Marchesi  ( https://github.com/pietromarchesi ) the original creator of these practicals.

# Requirements
In this practical, we will rely on using python, which is a popular programming language that makes it easy 
to write and understand code.
Furthermore, we will use Jupyter Notebooks, which is a special tool that lets you write and run Python code in small,
easy-to-read sections in your web browser. It’s like a digital notebook where you can mix text, 
(python-)code, and results all in one place. We have created one notebook for each practical, 
which you will download from this Gitlab page (see instructions below)

That means that you need to install Python and Jupyter Notebooks, and there are many ways to do so. 
If you are new to most of this, we recommend to install the [Anaconda Distribution](https://www.anaconda.com/download/) 
as this will make a lot of things easier for you.
Anaconda is a program that helps you easily manage Python and all the tools you might need for coding. 
It’s like a toolbox that comes with everything set up, so you don’t have to worry about installing things one by one. 
Anaconda includes Python, Jupyter Notebooks, and lots of useful packages for tasks like data science, 
machine learning, or scientific computing.
Note that the website asks you to register with your email, but you can also click on "skip registration". 
 
Once you have Anaconda installed, you need to create a python environment for this practical. A python environment is like a workspace where you can write and run Python code. Imagine it as a special box where you keep everything needed to work on a project. Inside the box, there’s the Python program itself, plus any extra tools (called packages or libraries) that help you with specific tasks like math, data analysis, or making websites. When you create an environment, a folder will be created on your computer, in which we will place all the code needed to run the practical. We have everything prepared for you and packed everything into a file, so you dont need to worry about how to set up the environment, if you follow the instructions below:

# Instructions
- Download Anaconda from the link given above, and be sure to select the Python 3.7 version.
- On this Github page ( https://github.com/franzkeaw/NMRC-practicals-2024 ), click on: Code (green button, top right)-> Download zip, and extract the zip file to a location on your laptop.
- Open Anaconda
- Go to the 'Environments' section
- Click 'Import'
- Select the path to the `nmrc2024.yml` file, which is in the zip file which you just downloaded, and choose a name for your environment (default will be `nmrc2024`).

Once your environment has been created:
- In Anaconda on the left you find "environments" -> Click on the environment name which you just created (a green triangle should appear) -> This will 'activate' your environment. This might take some time to load, and you know that your environment is activated when a green triangle appears next to it. [When you activate an environment in Python, you're telling your computer to use a specific workspace, to make sure that the right versions of Python and the right tools are used for that project. It's like stepping into a room where everything is set up just the way you need it for this practical. This includes the version of Python which we are using, and also additional code which is necessary to run the practical and which we have set up beforehand for you]
- Now we need to make sure that Jupyter Notebook is also added to our environment. After you activated your enviroment (previous step), click on 'home' (left side) in Anaconda, and look for Jupyter Notebook in the list of applications. If it has a button that says 'launch', then you can skip this step. Otherwise, the button should say 'install' <- Click on that one to make sure that Jupyter Notebook is added to your environment.
- Now navigate back to your list of environments (left side in Anaconda -> 'Environment') and click on the green triangle 
- Select 'Open with Jupyter notebook'. This will start a Jupyter notebook in your home directory, and it will open in your webbrowser.
- Within Jupyter Notebook which we just opened, navigate to the folder of the practicals (=the zipped and extracted folder which you downloaded from Github). 
    Note: If you are unable to navigate to your folder, you can also choose another option to open the jupyter notebooks:
    - Instead of choosing 'Open with Jupyter notebook' after clicking on the green triangle, choose 'open Terminal' and enter the following command (of course, replace the path in the command with the actual path to the folder which you downloaded):
    jupyter notebook --notebook-dir C:\FolderAAA\FolderBBB\FolderCCC\nmrc-practicals-updates_2024
    


In case you get 

```Widget Javascript not detected```

You may need to enable the widget extension with 

```$ jupyter nbextension enable --py --sys-prefix widgetsnbextension```

```$ jupyter nbextension enable --py widgetsnbextension```
