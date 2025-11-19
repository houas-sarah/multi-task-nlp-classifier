# Multi-Task NLP Classifier

This repository contains a multi-task NLP model trained on social media text.  
The goal is to predict, for a given sentence or tweet:

1. Emotion  
2. Type of gender-based violence (GBV)  
3. Hate / offensive / neutral label  

## Model overview

The model is a **shared BiLSTM** with three heads:

- A common text preprocessing pipeline (tokenisation, cleaning, padding)  
- A shared embedding layer and BiLSTM backbone  
- Three task-specific dense + softmax layers:
  - Emotion classification  
  - GBV type classification  
  - Hate / offensive / neutral classification  

The full training pipeline (data loading, preprocessing, model definition, training and evaluation) is implemented in `multi_task.ipynb`.  
The trained model and tokenizer are exported to the `models/` directory and used by the Streamlit app.


## How to run

### Install dependencies

Create and activate a virtual environment (optional), then:

pip install tensorflow streamlit scikit-learn pandas numpy matplotlib seaborn nltk


###  Run the notebook

Open `multi_task.ipynb` in Jupyter / VS Code and execute the cells to:

- load and preprocess the datasets,  
- train the multi-task model,  
- evaluate it on a validation set,  
- save the model and assets in `models/`.


###  Launch the Streamlit app

Once the `models/` folder contains `multitask_lstm.h5`, `tokenizer.json` and `label_maps.pkl`, run:
streamlit run app.py

The app provides a text box for input and returns the predicted:
- emotion  
- GBV type  
- hate/offensive label  


## Limitations

- The GBV dataset does not include a neutral or “no violence” class, so the model always assigns some form of violence, even for harmless sentences.  
- The hate/offensive task is trained on a specific dataset and may misclassify everyday, out-of-domain text.  
- The Streamlit app is a demonstration interface and should not be used as a real moderation or safety system.

## Possible extensions

- Add a true neutral / no-violence class with additional data  
- Try Transformer-based encoders instead of BiLSTM  
- Use class weighting or focal loss to handle imbalance  
- Add uncertainty estimation or confidence thresholds for safer use
