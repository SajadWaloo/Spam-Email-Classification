
# Spam Email Classification

This repository contains code for training a machine learning model to classify emails as spam or non-spam. The model is built using the Naive Bayes algorithm and utilizes the scikit-learn library.

## Dataset

The dataset used for training and evaluation is stored in the `fradulent_emails.txt` file. Each line in the file represents a single email, with the first column indicating the label (spam or non-spam) and the second column containing the email content. The dataset is preprocessed and converted into numerical features before training the model.

## Prerequisites

The following dependencies are required to run the code:

- numpy
- scikit-learn
- matplotlib

You can install the dependencies by running the following command:

pip install -r requirements.txt


## Usage

1. Clone the repository:
git clone https://github.com/your-username/spam-email-classification.git
cd spam-email-classification


2. Place the `fradulent_emails.txt` dataset file in the project directory.

3. Run the following command to execute the program:
python spam_classification.py


4. The program will load the dataset, preprocess the data, train the Naive Bayes model, make predictions on the test set, evaluate the model's performance, and generate a bar plot showing the distribution of predicted labels.

## Results

The program will display the confusion matrix and classification report, providing insights into the model's performance on the test set. Additionally, a bar plot will be generated to visualize the distribution of predicted labels.

## Contributing

Contributions to this repository are welcome. If you have any suggestions or improvements, please create a pull request or open an issue.
