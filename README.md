# Satya Bhaskar Chaliki
# 700769823


# Project: Question Answering & Conditional GAN

This project contains two assignments:

1. *Question Answering with Hugging Face Transformers*
2. *Digit-Class Controlled Image Generation using Conditional GAN*

---

##  *Question 1: Question Answering with Transformers*

###  Objective

Build a *Question Answering system* using Hugging Face’s transformers library. Learn to extract answers from a given context using pre-trained language models.

---

### Tasks & Implementation*

#### *Task 1: Basic Pipeline Setup*

* *Library*: transformers.pipeline
* *Steps*:

  * Load the default Question Answering model.
  * Provide a sample context and question.
* *Expected Output*:

  * Example answer: 'Charles Babbage'
  * Confidence score > 0.65
  * Valid start and end token indices

#### *Task 2: Use a Custom Pretrained Model*

* *Model Used*: deepset/roberta-base-squad2
* *Steps*:

  * Initialize the QA pipeline with the custom model.
  * Ask the same question using the same context.
* *Expected Output*:

  * Answer: 'Charles Babbage'
  * Confidence score > 0.70
  * Include start and end token indices

#### *Task 3: Test on Your Own Example*

* *Steps*:

  * Create a 2–3 sentence custom context.
  * Ask two different questions from the context.
* *Expected Output*:

  * Relevant answers to both questions.
  * score > 0.70 for both responses.

---

##  *Question 2: Digit-Class Controlled Image Generation with Conditional GAN*

###  Objective

Implement a *Conditional GAN (cGAN)* to generate MNIST digit images based on given class labels (0–9). This teaches how conditioning affects generative control.

---

###  *Tasks & Implementation*

#### *Task Description*

* Modify a basic GAN architecture to accept labels.
* *Label Embeddings*:

  * Concatenate label embedding with:

    * The noise vector → Generator input
    * The image → Discriminator input
* *Train*:

  * On MNIST dataset using PyTorch
* *Visualize*:

  * Generate one row of digits, 0 through 9 (1 image per label)

---

###  *Expected Output*

* A grid with one generated image per digit class (0–9)
* Generator learns to control output digit class from label input
* Loss values may fluctuate but image quality and label accuracy should improve across epochs

---

###  *Tools & Libraries*

* PyTorch, torchvision, matplotlib
* MNIST Dataset
* Optional: Use tqdm for progress bars during training

---

##  Project Structure

plaintext
project/
│
├── question_answering.py         # Task 1 implementation
├── conditional_gan.py            # Task 2 implementation
├── results/
│   └── generated_digits.png      # Visualization of conditional GAN output
├── README.md                     # This file
└── requirements.txt              # pip dependencies
