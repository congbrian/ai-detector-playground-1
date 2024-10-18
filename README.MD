
initial playground for:

Developing an LSTM that utilizes cross-perplexity measurement in order to classify AI vs Human texts.

sources 
main source: [Binoculars](https://github.com/ahans30/Binoculars)
auxiliary brute force source: [ghostbuster](https://github.com/vivek3141/ghostbuster)
dataset target: subsection of [RAID](https://github.com/liamdugan/raid) training data.


primary goal of this repo is:
1. Copy Binoculars and run it on RAID data (working with a subset of raid)
2. create our own implementation of Binoculars retrieving per token perplexity and cross perplexity rather than a median value for each entry.
3. Run our code on (a subset of ) the RAID data. Return the scores.
4. Train an LTSM on the scores.
5. Compare it to Binoculars performance and accuracy-wise.
