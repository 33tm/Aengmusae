# Parrot

A demo is available at [tttm.us/parrot](https://tttm.us/parrot).

Instructions for how to train/run the model are at the bottom.

## Reversing Korean romanization with AI

Romanizing Korean has always been inconsistent because of the lack of consensus between the [many different Korean romanization systems](https://en.wikipedia.org/wiki/Romanization_of_Korean#Systems). In addition, the language is difficult to romanize due to how many words have the same pronunciation, yet have spelling differences that are difficult to express with the english alphabet. This prevents something like [Pinyin](https://en.wikipedia.org/wiki/Pinyin) from existing.

Since 2000, the [Revised Romanization of Korean](https://en.wikipedia.org/wiki/Revised_Romanization_of_Korean) has been the official romanization system of South Korea. Even with an official standard, non-standardized romanization is common due to the lack of enforcement/care outside of government related material. Specifically, I noticed how local businesses such as restaurants and small stores each had their own quirks in terms of romanizing their names. 

Interested in seeing how a neural network would interpret the different romanization methods and create predictions, I created an LSTM RNN model trained on romanized/korean word pairs. I didn't have high hopes for this model at all--mostly due to my inexperience in this whole field--but also due to how the model completely ignores the issue of words with the same pronunciation having different spellings. It discards any context provided from the original sentence. Even if the dataset is created from full sentences, I made the decision to split the sentence into individual English-Korean word pairs in hope to make the learning process for me easier.

### Dataset

I scraped romanized/korean lyrics from [Color Coded Lyrics](https://colorcodedlyrics.com) in hope that community romanization would have more variability in its romanization and possibly have unstandardized patterns the model could learn off of.

Additionally, I got more romanized/korean pairs from the Korean dictionary from [Kaikki](https://kaikki.org/dictionary/Korean/), which was created using [Wiktextract: Wiktionary as Machine-Readable Structured Data](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.140.pdf). Using this dictionary in this case could have been a mistake since presumably something like this would have used a single romanization system, reducing the model's overall ability to generalize romanization patterns.

In total, I got 465,247 Korean-English word pairs to train on.

[`train.csv`](https://github.com/33tm/Parrot/releases/download/model/data.csv)

### Issues

The model does not address the issue of different words having the same pronunciation. As mentioned above, the training data is split into individual words and the sentences (and the context of the word) are discarded. Since Korean is a language that depends on the context of a word to know its spelling just from hearing it, this seemingly minor issue makes this model essentially useless in terms of reliably getting the correct Korean word from a romanized input.

Although the dataset had varied romanization standards incorporated, it was still incredibly biased towards the [Revised Romanization of Korean](https://en.wikipedia.org/wiki/Revised_Romanization_of_Korean), the official standard, and often returned incorrect Korean for words that were not formatted in said standard.

I also noticed how the model failed to predict the last syllable of a word correctly unless you repeated the last word multiple times.

```
Expected: nan mwonga dalla dalla => 난 뭔가 달라 달라

Input: nan mwonga dalla dalla
Output: 난 뭔가 달ㄹ 달ㄹ
                       .      .
Input: nan mwonga dallaa dallaa
Output: 난 뭔가 달라 달라
```
```
Expected: jagiya => 자기야

Input: jagiya
Output: 자기
             .
Input: jagiyaa
Output: 자기ㅇ
             ..
Input: jagiyaaa
Output: 자기야
```

I'm sure this is an incredibly trivial issue related to the input tensors, I'll make sure to come back and take another look at it once I get more experience working with this ecosystem :D

### Possible Improvement

I really think this model could be more useful if it was trained with context (on the original sentences instead of words) so that this model could actually be useful when turning large amounts of unstandardized romanized Korean text to Korean.

### What I learned

- Generalizing human-made content is very hard
- I need more data (look mom I'm becoming a large corporation)

## How to Run

You can download the dataset from [`data.csv`](https://github.com/33tm/Parrot/releases/download/model/data.csv) and skip the data collection process.

You can download the model from [`model.pt`](https://github.com/33tm/Parrot/releases/download/model/model.pt) and skip the training process.

Put the files in `/out`.

Make sure to download both to run the API.

### Data Collection
```bash
# Install dependencies
pip3 install bs4 requests

# Run data collection
python3 data.py
```
This creates a [`data.csv`](https://github.com/33tm/Parrot/releases/download/model/data.csv) file in `/out`. Takes quite a while to scrape, I recommend just downloading the premade dataset if you want to train the model.

### Training
Training requires [`data.csv`](https://github.com/33tm/Parrot/releases/download/model/data.csv) to be in `/out`.

Install torch [for your system](https://pytorch.org/get-started/locally/)
```bash
# Run your specific torch installation command
# e.g. pip3 install torch

# Install other dependencies
pip3 install jamo

# Train the model
python3 model.py
```
Make sure to use any acceleration you have, this takes quite a bit of time.

| GPU                     | Time (24 epochs) | Average time/epoch   | Rental price |
|-------------------------|------------------|----------------------|--------------|
| NVIDIA GeForce GTX 980  | 9h 42m 43s       | ~24 Minutes          | N/A          |
| NVIDIA GeForce RTX 4090 | 39m 36s          | ~1 Minute 30 Seconds | ~$0.26/hour  |
| NVIDIA H100             | 33m 44s          | ~1 Minute 20 Seconds | ~$2.80/hour  |

(The rental H100 was severely bottlenecked by its processor so it probably would have been faster)

The RTX 4090 definitely had the best value to rent, especially for a small project like this. Poor GTX 980 probably cost more in electricity lol :sob:

The training process creates a [`model.pt`](https://github.com/33tm/Parrot/releases/download/model/model.pt) file in `/out`.

### API
Running the API requires the same dependenices as training the model. It requires both [`data.csv`](https://github.com/33tm/Parrot/releases/download/model/data.csv) and [`model.pt`](https://github.com/33tm/Parrot/releases/download/model/model.pt) to exist in `/out`.

```bash
# Install dependencies (+training dependencies)
pip3 install flask flask_cors waitress

# Run the API
python api.py
```
This runs a server on port 8080. To make a query, send a POST request to `/` with a JSON body with `query` set to your input (romanized korean).

`POST localhost:8080 { "query": "aengmusae" }`

=> `앵무새`
___
Thank you so much for reading!