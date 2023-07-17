# Instructions

## Download

Clone this repository from GitHub (HTTPS)

```shell
git clone https://github.com/zachgordon25/skeleton-app.git
```

Clone this repository from GitHub (SSH)

```shell
git clone git@github.com:zachgordon25/skeleton-app.git
```

## Setup

Change directory into the project

```shell
cd skeleton-app
```

### Setup virtual environment

Create a Virtual Environment

```shell
python -m venv .venv
```

Activate Virtual Environment (Mac)

```shell
source .venv/bin/activate
```

Activate Virtual Environment (Windows)

```shell
source .venv/Scripts/activate
```

Upgrade PIP (Mac)

```shell
pip install --upgrade pip
```

Upgrade PIP (Windows)

```shell
python.exe -m pip install --upgrade pip
```

### Install Python packages

Install required Python packages

```shell
pip install -r requirements.txt
```

### Run the app
```shell
flask run
```

### Run the app in debug mode
```shell
python app.py
```