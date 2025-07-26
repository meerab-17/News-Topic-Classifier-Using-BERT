# News-Topic-Classifier-Using-BERT

objectives:

The goal of this project is to fine-tune a pre-trained transformer model (like BERT) to automatically categorize news headlines into their respective topics.

Dataset:
I am using the AG News Dataset, which is readily available through the Hugging Face datasets library. It contains news headlines along with their corresponding topic labels.

1. Environment Setup
Begin by installing the necessary Python libraries:

transformers

datasets

torch

scikit-learn

streamlit or gradio (for deployment)

-Load the AG News dataset using the Hugging Face Datasets library.

-Explore the structure and content, including how headlines are mapped to labels.

-Tokenize the text data using the bert-base-uncased tokenizer.

-Convert the categorical labels into numerical format.

-Split the dataset into training and testing sets for evaluation.

-Load the bert-base-uncased model configured for sequence classification.

-Define training parameters like batch size, learning rate, and number of epochs.

-Fine-tune the BERT model using the Hugging Face Trainer API.

-Track training performance such as loss and accuracy.

-Evaluate the fine-tuned model on the test data.

-Use accuracy and F1-score as performance metrics via sklearn.metrics.

-Develop a lightweight web interface using either Streamlit or Gradio.

-Users can input a news headline and receive a predicted topic in real time.

conclusion:
I was able to implement practical NLP using transformer models, transfer learning and fine-tuning models, understand and apply evaluation matrices and deploy machine learning models as web apps.
This task involves building a text classification model using the AG News dataset. A pre-trained BERT model (bert-base-uncased) is fine-tuned to predict the topic of a news headline. The process includes data preprocessing, model training, evaluation using accuracy and F1-score, and deployment via a simple Streamlit interface for live user interaction.

