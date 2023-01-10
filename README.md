## dodge

### ./models/

Implementations of various transformer based models for League of Legends Match prediction. A model can trained by executing the command ```python models/run.py combined --db_path data/raw/matches.db --epochs 20```. See the ./models/run.py file for a full list of argument options. The SQLite database should have the same structure my [opgg-webscraper](https://github.com/loua19/opgg-scrape) produces. I have included a small .csv file of euw matches in the correct format, you may run data/build_db.py to build this file into a .db file compatible with SQLite. 

 From my testing, the models in their current form are capable of predicting the outcome of a high-elo League of Legends match with ~63% accuracy on a held out test set. Unlike most other attempts I have come across, these models appropriately mask each players match history, preventing it from using information it would not have access to at test time (unlike [this](https://arxiv.org/pdf/2108.02799.pdf) paper for example).

### ./notebooks/group_match_prediction_paper

An implementation of the paper [Group Match Prediction via Neural Networks](https://ceur-ws.org/Vol-2960/paper1.pdf) for predicting the outcome of a match by modelling champion and player-champion 'usefulness' values. Overall I found that the proposed model develops some understanding of what makes a good team composition. The results, however, don't exceed that of other common models applied to the problem. More specifically, my transformer models and models that take advantage of pre-calculated statistics (champion win rate, player experience) [[Link]](https://arxiv.org/pdf/2108.02799.pdf)] give better results by a wide margin.

### Installation

You may install the necessary dependencies using conda as follows:

- cd to dodge/ and run the following commands
- ```conda create --name dodge --file ./config/req.yml```
- ```conda activate dodge```

 Note that I have encountered some strange behaviour when using the non-CUDA version of PyTorch (issues with dtypes in dataset.py). If you are going to install the dependencies yourself, I recommend making sure the install the CUDA version of PyTorch. 