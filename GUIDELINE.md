# **How to contribute?**

Instruction:
1. Install [Git](https://git-scm.com/) or [GitHub Desktop](https://desktop.github.com/), and an editor such as Visual Studio Code.

2. Clone the codes to your computer (**qoop** is a core package of our various repository)
```
git clone https://github.com/vutuanhai237/UC-VQA.git
cd UC-VQA
rm -rf qoop (delete folder qoop)
git clone https://github.com/vutuanhai237/qoop.git
```

Note that the **qoop** folder must be on the same level as the **codes** folder.

3. Make sure that you have installed python 3+ version. After that, install all needed packages.
```
pip install -r requirements.txt
```
4. Test

Run all test case
```
cd tests
pytest
```