# Data-English-Chinese-Conversion

#### English to Chinese Translation

Please run alpaca_data_cleaning.py

Using two A30 GPU, it took us 35 minutes to translate the first 500 pieces of data.

#### Translation Results

Check alpaca_cleaned_chinese_qwen.json. 

It contains the translation results of the first 500 pieces of data in the alpaca - cleaned dataset.

#### Evaluation

Please run conver_eval.py

Analyze the proportion of English in each piece of data. First, concatenate the contents of instruction, input, and output in the data. Then, if the proportion of English in this text exceeds 8%, the translation of this piece of data is considered substandard; if it is less than 8%, the translation of this piece of data is considered up to standard. Finally, calculate the proportion of up - to - standard data among all the data as the final evaluation score.

The translation accuracy of our evaluation on the first 500 pieces of data is 80%.
