1. AUTHORS :
Romain Boyer
Flore de Lasteyrie

2. FINAL SYSTEM :

The big steps of our final system are : 
- Importing data
- Modifying the sentences + the stop words to keep all the negation words 
  (no/nor/not)
- Cleaning the data (tokenising, is alpha, is not stop word)
- Stemming the data
- Creating a BOW vector
- Transforming this vector with TF-IDF scores
- Applying a Linear Support Vector Classifier (the one that gave the best results)

Some results :
MultinomialNB  # With alpha=0.53  -> acc=78.72
LinearSVC      # With C=0.138     -> acc=79.26

In src you can find 3 classifier python files :
- classifier_spacy which uses the SpaCy library and lemmatisation 
  but has a very long computation time
- classifier_dev which was used during development to test different algorithms,
  visualise accuracy scores and confusion matrices (on the train and the test set)
- classifier which is the “production” file; no print and only the chosen algorithm
  to be as clear as possible

3. ACCURACY ON DEVDATA
79.26