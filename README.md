# Project on Reinforcement Learning: Proximal Policy Distillation (PPD)
### Course Project for **Machine Learning and Deep Learning (MLDL)**
### Institution: **Politecnico di Torino**
### Teaching assistants: Andrea Protopapa and Davide Buoso

The official assignment document is available here: [Google Doc](https://docs.google.com/document/d/16Fy0gUj-HKxweQaJf97b_lTeqM_9axJa4_SdqpP_FaE/edit?usp=sharing).

---

## Project Overview

This project focuses on **Proximal Policy Distillation (PPD)**, a Reinforcement Learning method that enhances the stability of **Proximal Policy Optimization (PPO)** with a distillation loss.

* **Objective:** To transfer knowledge from a robust, pre-trained teacher policy to a student policy to accelerate learning and improve robustness.
* **Environment:** The experiments were conducted on a custom version of the **Gym Hopper** environment, which utilizes **Uniform Domain Randomization (UDR)** to improve the generalization capabilities of the teacher policy.
* **Methodology:** Policy distillation is performed onto three student network sizes (smaller, identical, and larger than the teacher's) to compare PPD's performance against standard PPO without distillation.

For a detailed analysis, results, and discussion, please refer to the full project report: [report_ML_PPD.pdf](report_ML_PPD.pdf).



---
In order to be able to run the code on your computer follow these instructions.

### 1. Local on Linux (recommended)

If you have a Linux system, you can work on the course project directly on your local machine. By doing so, you will also be able to render the Mujoco Hopper environment and visualize what is happening. This code has been tested on Linux with python 3.7.

**Installation**
- (recommended) create a new conda environment, e.g. `conda create --name mldl pip=22 python=3.8 setuptools=65.5.0 wheel=0.38`
- Run `pip install -r requirements.txt`
- Install MuJoCo 2.1 and the Python Mujoco interface:
	- follow instructions here: https://github.com/openai/mujoco-py
	- see Troubleshooting section below for solving common installation issues.




### 2. Local on Windows
As the latest version of `mujoco-py` is not compatible for Windows explicitly, you may:
- Try installing WSL2 (requires fewer resources) or a full Virtual Machine to run Linux on Windows. Then you can follow the instructions above for Linux.
- (not recommended) Try downloading a [previous version](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/) of `mujoco-py`.


Check your installation by launching `python test_random_policy.py`.

---

## Troubleshooting
- General installation guide and troubleshooting: [Here](https://docs.google.com/document/d/1j5_FzsOpGflBYgNwW9ez5dh3BGcLUj4a/edit?usp=sharing&ouid=118210130204683507526&rtpof=true&sd=true)
- If having trouble while installing mujoco-py, see [#627](https://github.com/openai/mujoco-py/issues/627) to install all dependencies through conda.
- If installation goes wrong due to gym==0.21 as `error in gym setup command: 'extras_require'`, see https://github.com/openai/gym/issues/3176. There is a problem with the version of setuptools.
- if you get a `cannot find -lGL` error when importing mujoco_py for the first time, then have a look at my solution in [#763](https://github.com/openai/mujoco-py/issues/763#issuecomment-1519090452)
- if you get a `fatal error: GL/osmesa.h: No such file or directory` error, make sure you export the CPATH variable as mentioned in mujoco-py[#627](https://github.com/openai/mujoco-py/issues/627)
- if you get a `Cannot assign type 'void (const char *) except * nogil' to 'void`, then run `pip install "cython<3"` (see issue [#773](https://github.com/openai/mujoco-py/issues/773))
