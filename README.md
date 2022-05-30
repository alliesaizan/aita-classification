# A BERT-based text classifier!
![](static/icon.png "AITA Icon")

## Table of Contents
1. Overview
2. How to run this experiment on your own machine
3. Development process
4. Lessons learned

# Overview
Welcome! This application is a small experiment that uses the [BERT large language model](https://huggingface.co/distilbert-base-uncased) to classify posts from
the [Am I The Asshole (AITA) subreddit](https://www.reddit.com/r/AmItheAsshole/). The model used to generate predictions is actually a small, fast, "distilled" version of the BERT model meant for finetuning ondownstream tasks. Please note, some of the files needed to be zipped to accomodate Git Large File Storage. The raw data [folder](data/raw) and [trained model](results/) are zip files. You can find the original project structure in my [OneDrive folder](https://1drv.ms/u/s!AkUOTbaWXaF8gbtE71qCMCRxTM3rvQ?e=RhnEYH) (view-only!). I've also posted this summary to my [personal blog](https://alliesaizan.github.io/)

The repository is structured as follows:

- data
  - raw
    - big-query-aita-aug18-aug19.zip; The results of a SQL query run against a database of reddit posts. The dataset covers all AITA posts from August 2018-2019. Unzip this file to use it. 
- results
  - checkpoint-20238.zip --The final PyTorch text classification model, saved at the last epoch runtime. Unzip this file to use it.
 
