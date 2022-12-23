## dodge

Implementations of various neural models for League of Legends Match prediction.

# ./notebooks/champ_only

This includes a variety of models (MLP, LSTM, ect..) trained exclusively on team compositions. Overall I found these models are very sensitive and very quickly over-fit the data (when given enough parameters).

# ./notebooks/group_match_prediction_paper

An implementation of the paper [Group Match Prediction via Neural Networks](https://ceur-ws.org/Vol-2960/paper1.pdf) for predicting the outcome of a match by modelling champion and player-champion 'usefulness' values. Overall I found that the proposed model develops some understanding of what makes a good team composition. The results, however, don't exceed that of other common models applied to the problem. More specifically, models that take advantage of pre-calculated statistics (champion win rate, player experience) [[Link](https://arxiv.org/pdf/2108.02799.pdf)] still give better results by a wide margin.

# extra stuff

The repository also includes data crawling/processing for League of Legends match data using the Riot API and Pandas. This can be found in the ./src directory. If you would like to use this simply install the relevant libraries (in particular, [Riot Watcher](https://github.com/pseudonym117/Riot-Watcher) is a dependency), and modify/run main.py (found in src). You must also store your API key in an environment variable named RIOT_API_KEY. Alternatively, you may follow the instructions below to create a copy of my personal conda environment for the project (note recommended). Please note that this environment includes PyTorch/CUDA and therefore will take a long time to download/install.

- clone the repository to your local machine.
- cd to dodge/ and run the following commands
- ```conda create --name dodge --file ./config/req.txt```
- ```conda activate dodge```
- ```conda env config vars set RIOT_API_KEY=<your_api_key>```