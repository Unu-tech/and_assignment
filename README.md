# Setup

Clone the repository locally:

```bash
git clone git@github.com:Unu-tech/and_assignment.git
```

<details>
<summary> When setting up GCP VM from scratch </summary>

Setting up VM from scratch requires some additional installs. Assuming linux machine, do the following:

```bash
sudo apt-get update && sudo apt upgrade
# Following are needed for python installation with pyenv (may be redundant).
sudo apt-get install bzip2 libncurses5-dev libncursesw5-dev libffi7 libffi-dev \
 libreadline8 libreadline-dev zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev lzma liblzma-dev libbz2-dev

```

If there's an error due to version numbering, search for latest version. 

```bash
# E: Unable to locate package libreadline5

apt-cache search libreadline
```

</details>

## Local dev environment

### Step 1: Install pyenv 

**python local environment is managed by pyenv**. Follow the instructions [here](https://github.com/pyenv/pyenv). 

Once the installation on you machine done, install python version.

```bash
# install Python 3.11.11
pyenv install 3.11.11

# create new virtual environment (choose any name you like)
pyenv virtualenv 3.11.11 assign

# On the root of the repository, set the new environment as the local environment
pyenv local assign
```

### Step 2: install dependencies

```bash
make install   # install dependencies
```

# Training

Please feel free to check relevant configs and `train.sh` to understand how to use. But the easiest way is sourcing `train.sh` which creates logs as well.

```bash
source train.sh
```
