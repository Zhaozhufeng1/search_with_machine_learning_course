
# Split labeled data into training and test.
head -n -10000 /workspace/datasets/fasttext/output.fasttext > phone.train
tail -10000 /workspace/datasets/fasttext/output.fasttext > phone.test

# Train model
~/fastText-0.9.2/fasttext supervised -input phone.train -output model_phone

# Test model for P@1 and R@1
~/fastText-0.9.2/fasttext test model_phone.bin phone.test

# Test model for P@5 and R@5
~/fastText-0.9.2/fasttext test model_phone.bin phone.test 5


# Increase number of epochs to 25
~/fastText-0.9.2/fasttext supervised -input phone.train -output model_phone -epoch 25
~/fastText-0.9.2/fasttext test model_phone.bin phone.test

# Increase number of epochs to 100
~/fastText-0.9.2/fasttext supervised -input phone.train -output model_phone -epoch 100
~/fastText-0.9.2/fasttext test model_phone.bin phone.test

# Increase the learning rate to 1.0 and go back to 25 epochs
~/fastText-0.9.2/fasttext supervised -input phone.train -output model_phone -lr 1.0 -epoch 25
~/fastText-0.9.2/fasttext test model_phone.bin phone.test

# Set word ngrams to 2 to learn from bigrams
~/fastText-0.9.2/fasttext supervised -input phone.train -output model_phone -lr 1.0 -epoch 25 -wordNgrams 2
~/fastText-0.9.2/fasttext test model_phone.bin phone.test








