# Project 1 -  Navigation

## Introduction
For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

![Bananaworld](p1-navigation/images/bananaworld.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Setting up
Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

### Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

Then place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and then write the correct path in the argument for creating the environment under the notebook `Navigation_solution.ipynb`:
```python
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
```

## Description of files
* `dqn_agent.py`: Implementation of the agent.
* `model.py`: Implementation of the Q-network.
* `dqn.pth`: Model weights for a pretrained DQN model.
* `ddqn.pth`: Model weights for a pretrained Double DQN model.
* `Navigation.ipynb`: Explore the unity environment.
* `Navigation_solution.ipynb`: A possible implementation of how to train an agent.
* `Enjoy_agent.ipynb`: A notebook allowing you to watch pretrained models.
* `Navigation_Pixels.ipynb`: An optional extra challenge (See describtion below).

## Instructions 
Run the `Navigation_solution.ipynb` to train your own agent. 
Run the `Enjoy_agent.ipynb` to watch a trained agent. If watching your own agent, make sure that the setup is the same as when you trained the agent.
