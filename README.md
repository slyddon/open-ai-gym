# open-ai-gym

## Dev environment

To set up the appropriate environment run the following commands:
```
conda create --prefix ./.env python=3.6.8 --yes
conda activate ./.env
conda install -c pytorch pytorch --yes
conda install -c conda-forge numpy matplotlib jupyterlab --yes
pip install gym watermark
```

## Directory structure

```
├── README.md             <- The top-level README for developers using this project
├── .gitignore            <- .gitignore configuration file
├── src                   <- Folder containing python scripts
│   ├── action_bot        <- Folder containing action bot scripts
│   ├── dqn               <- Folder containing deep Q network scripts
│   ├── episode_memory    <- Folder containing episode memory scripts
│   ├── q_table           <- Folder containing Q table scripts
│   ├── replay_memory     <- Folder containing experience replay memory scripts
│
├── notebooks             <- Folder containing notebooks for running open-ai-gym
├── fitted_models         <- Folder for storing fitted network models
```
