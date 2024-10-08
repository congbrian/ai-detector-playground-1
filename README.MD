initial playground for
"
1st goal, and I think the simplest: Develop an LSTM that utilizes cross-perplexity measurement in order to classify AI vs Human texts, potentially multi-class classification to account for different adversarial attacks.
"

sources 
binoculars: https://github.com/ahans30/Binoculars
ghostbuster: https://github.com/vivek3141/ghostbuster
dataset target: RAID training data.


primary goal of this repo is:
1. copy binoculars and run it on RAID data (working with a subset of raid)
2. copy ghostbuster and get it to run on RAID data (temp stall on this as the datasets this model were trained on are deprecated openai and therefore incures cost and likely need to be retrained).
3. create our own implementation of cross-perplexity based on binoculars.
4. get code to run
5. Run our code on (a subset of ) RAID data
6. compare it to the other two. 
