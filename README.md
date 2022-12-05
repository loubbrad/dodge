## dodge

An implementation of the paper [Group Match Prediction via Neural Networks](https://ceur-ws.org/Vol-2960/paper1.pdf) for analysing League of Legends game data using neural networks (PyTorch). Overall I found that the model proposed does develop some understanding of what makes a good team composition. The results,  however, don't exceed that of other common models applied to the problem.

The repository also includes data crawling/processing for League of Legends match data using the Riot API and Pandas. If you would like to use this simply install the relevant libraries (in particular, [Riot Watcher](https://github.com/pseudonym117/Riot-Watcher) is a dependency), and modify/run main.py (found in src). You must also store your API key in an environment variable named RIOT_API_KEY. Alternatively, you may follow the instructions below.

- clone the repository to your local machine.
- modify ./config/environment.yml to include your API key.
- modify main.py for your use case.
- install anaconda/conda
- cd to dodge/ and run the following commands
- ``` conda env create -f config/environment.yml ```
- ``` conda activate dodge ```
- ``` python -u "./src/main.py" ```
