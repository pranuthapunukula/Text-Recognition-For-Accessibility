# Text Recognition For Accessibility

## Introduction

**Background**

The goal of our project is to promote accessibility for people with visual impairments by identifying text from a picture and displaying this text in the form of an indicator. The model will point out significant text in the picture and display the text in a way that is easily identifiable and can be recognized using a voice to text system. Given that the world is becoming increasingly digital, it is important to implement systems that can extract text from images in an efficient manner given that images are becoming a key signifier for identification and health related issues. This project is important because improving accessibility provides equal access to people with visual disabilities. The potential that text extraction from images has for the safety of those with visual impairments proves the importance of this project, and can be used in a number of ways. Some uses can include identifying one's surroundings or gathering information about certain ingredients in products on a store shelf. The research we conduct from this project ensures that all people can perceive, understand, and navigate the world equally and with assistive technology when needed. 

**Objective**

The dataset we will be using is the Kaggle Text Extraction from Images Dataset. This dataset provides images that are effective examples of recognizable text in images in order to predict the efficiency of the model itself. The examples in the model dataset help provide an effective manner of predicting the error rate regarding the correctness of the text that is extracted from the image. The dataset consisted of 25,000 images that included 1 million annotations. However, for the purpose of this project, the data was narrowed down to 2,000 data points that were evaluated on a smaller scale. Our proposed method for building a product is using a Neural Network to identify text from images using Natural Language Processing. In this project, the concept of OCR was put into practice in order to digitize the physical character aspects that are able to be extracted from the images themselves. 

**Audience**

The prospective audience for this machine learning project are those who suffer from a vision impairment or have diagnosed blindness. The project would target those who could use images as a way of navigating the physical work through the perception of a digitized image. Those who would benefit from this project would most likely use the text extraction tool with a text-to-voice tool that allows the text to be heard and perceived accordingly. This project may also be beneficial for those who would rather read the ALT text interpretations of images or have trouble reading the text on a smaller scale rather than in a format that brings attention to the extracted text.  The purpose of the project is to create a machine learning model that allows users who suffer from a vision impairment to be able to efficiently read the text that comes from the images. This method will work closely with other assistive tools to efficiently provide a perspective of the digital world to those who need to use images to perceive their environment. This project will make it more efficient for those who have visual impairments to be able to navigate digital environments using a voice to text system that includes pictures. Text from images can be used in a number of ways to create more inclusive environments for those who can use this tool to the advantage of recognition and predictive capabilities. It can also be used as a tool to those with learning or cognitive disabilities and can provide adequate assistance to those who are recovering from a temporary disability. 

 
## Dataset

**Data Description**

The dataset we used was from Kaggle, and consisted of images and word annotations for people to implement optical character recognition models with. The dataset was very large, coming in at 7 gigabytes, with over 25,000 photos. The data came with training and test images, as well as an annotations dataset to train the model with, too. These were in the form of JSON and Parquet-type files. The data was collated by Kaggle user: Rob Mulla (Owner) and the dataset is available with a CC0: Public Domain License. 

**Processing**

Because of the size of the data, we had to downsize it to make it easier to work with. We downsized the annotations data to 1000 samples, compared to the original million+. Additionally, we merged two datasets, image_df and annot_df, on the basis of the image_id column. 

<img width="500" alt="Screen Shot 2023-05-01 at 5 45 25 PM" src="https://user-images.githubusercontent.com/51467244/235544429-1a35bf93-0177-48df-b027-9dea99d645a2.png">

```
# Importing Libraries

import pandas as pd
import numpy as np

from glob import glob
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from PIL import Image

plt.style.use('ggplot')
```

**Plotting Example Images**

```
# Plotting example images

img_fns = glob.glob('../Final_Project/train_val_images/train_images/*')
fig, axs = plt.subplots(5, 5, figsize=(20, 20))
axs = axs.flatten()

# Displaying the first 25 images

for i in range(25):
 axs[i].imshow(plt.imread(img_fns[i]))
 axs[i].axis('off')
 image_id = img_fns[i].split('/')[-1].rstrip('.jpg')
 n_annot = len(annot_df.query('image_id == @image_id'))
 axs[i].set_title(f'{image_id} - {n_annot}')
 
plt.show()
```

<img width="600" alt="Screen Shot 2023-04-25 at 4 02 38 PM" src="https://user-images.githubusercontent.com/51467244/235227359-653ba86a-952e-481b-858a-a1aa5db9f6d7.png"> 

```
image_id = img_fns[0].split('/')[-1].split('.')[0]
annot.query('image_id == @image_id')
```

<img width="600" alt="Screen Shot 2023-04-28 at 2 14 26 PM" src="https://user-images.githubusercontent.com/51467244/235233716-d04d8756-d4e7-4a4b-823f-49a403003e52.png">



## Methodologies

**Extracting Text from Images using Keras OCR**

We used Keras-OCR, a deep-learning tool, to analyze the text in the images. OCR’s in general are used to identify and digitize characters in images. Generally, Keras-OCR works in parts, including: Image Processing, Character Segmentation and Recognition, and Text Recognition. The Image Processing is relatively straight-forward, in that Keras-OCR ingests images from the dataset and processes it to make text more visible and blur any background to further enhance it. Character segmentation is achieved by splitting each individual character in a string to further analyze it. Keras-OCR does this by implementing contour detection, which is a technique that detects the borders of objects by finding differences in shading, and localizing them for analysis. 

After segmenting the characters, Keras-OCR can begin Recognition by using deep learning models, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs) to achieve this. These models in Keras-OCR are pre-trained on large datasets of labeled images to identify each character with relative accuracy. Lastly, Keras-OCR began Text Recognition, which entails stringing the characters that have been split and recognized, into words and sentences. This is achieved by using Natural Language Processing models. From there, Keras-OCR also has a function in which it visualizes the words, which makes it simple to analyze the accuracy of the models.


<img width="500" alt="Screen Shot 2023-04-28 at 2 06 17 PM" src="https://user-images.githubusercontent.com/51467244/235232311-6bdf6d2e-2870-4de7-98f6-f9757eaa7703.png">

*VGG illustration - U Toronto's Prof Davi Frossard*

The core of the Keras-OCR package has two parts: a core detector and a recognizer. These provide wrappers for the underlying models. However, they also offer a pipeline you can use to run on images. Within the pipeline, you have options to adjust the parameters for the detector, recognizer, scale, and max size. When running the pipeline, it provides the user with pre-trained weights that are used to make the predictions. Using the pipeline recognize command, we can run through a list of images and ask it to make predictions. The table results Keras OCR offers includes the predicted word and its bounding box. By wrapping the results into a dataframe, we were able to format the results in a way that could be plotted. Keras OCR automatically has a built in annotation drawing tool, so we were able to visualize the predicted results directly on the images. 

```
# Installing Keras OCR (supports Python >= 3.6 and TensorFlow >= 2.0.0)

import keras_ocr

# Setting up a pipeline with Keras-ocr (the model is a pre-trained text extraction model loaded with pre-trained weights for the detector and recognizer)
# Plotting annotations for the first image

pipeline = keras_ocr.pipeline.Pipeline()
results = pipeline.recognize([img_fns[1]])
keras_ocr.tools.drawAnnotations(plt.imread(img_fns[1]), results[0])

pd.DataFrame(results[0], columns=['text', 'bbox'])
fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(plt.imread(img_fns[1]), results[0], ax=ax)
ax.set_title('Keras OCR Result Example')

plt.show()

```
<img width="600" alt="Screen Shot 2023-04-28 at 1 40 05 PM" src="https://user-images.githubusercontent.com/51467244/235227638-1623994a-f0af-4a7d-a64f-7204f9d2be59.png">

```

# Setting up a pipeline with Keras-ocr (the model is a pre-trained text extraction model loaded with pre-trained weights for the detector and recognizer)
# Plotting annotations for all images

results = pipeline.recognize([img_fns[1]])
pd.DataFrame(results[0], columns=['text', 'bbox'])
fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(plt.imread(img_fns[1]), results[0], ax=ax)
ax.set_title('Keras OCR Result Example')
plt.show()

# Running the pipeline recognizer on the first 25 images and making predictions about the text in these images

pipeline = keras_ocr.pipeline.Pipeline()
dfs = []
for img in tqdm(img_fns[:25]):
results = pipeline.recognize([img])
result = results[0]
img_id = img.split('/')[-1].split('.')[0]
img_df = pd.DataFrame(result, columns=['text', 'bbox'])
img_df['img_id'] = img_id
dfs.append(img_df)
kerasocr_df = pd.concat(dfs)

# This ended up being unecessary since we weren't comparing OCR's, but plotting comparisons

def plot_compare(img_fn, kerasocr_df):
img_id = img_fn.split('/')[-1].split('.')[0]
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
keras_results = kerasocr_df.query('img_id == @img_id')[['text','bbox']].values.tolist()
keras_results = [(x[0], np.array(x[1])) for x in keras_results]
keras_ocr.tools.drawAnnotations(plt.imread(img_fn),
keras_results, ax=axs[1])
axs[1].set_title('keras_ocr results', fontsize=24)
plt.show()
for img_fn in img_fns[:25]:
plot_compare(img_fn, kerasocr_df)

# Printing the identified text from the images

predicted_image = prediction_groups[1]
for text, box in predicted_image:
    print(text)
    
```
<img width="600" alt="Screen Shot 2023-04-28 at 1 43 52 PM" src="https://user-images.githubusercontent.com/51467244/235228287-fbb6dccd-f1e0-42e3-9873-ddd804028902.png">

<img width="300" alt="Screen Shot 2023-04-25 at 4 29 20 PM" src="https://user-images.githubusercontent.com/51467244/235227852-025e3cf1-774f-43a8-8ff7-23fbeb83089c.png"> <img width="300" alt="Screen Shot 2023-04-25 at 4 29 14 PM" src="https://user-images.githubusercontent.com/51467244/235227864-9a412ac0-1fd3-4eef-bb8b-dd5f1f4133ee.png">



**Generating Metrics -** ***How well does our model perform?***

To evaluate the model, we used two corresponding metrics: Word Error Rate (WER) and Character Error Rate (CER). These two models identify the performance of the OCR system that is used to extract the text from the images and together they represent the system’s ability to correctly recognize the text. WER focuses on the amount of words the system incorrectly recognizes and this metric is used to determine the faults within each extraction regarding the credibility of the text. It is calculated using the formula WER = (S + D + I) / N. In this formula, S is the number of word substitutions the system makes that are incorrectly matched with the word that is presented in the image. D is the number of word deletions the system makes within an image text extraction and I is the number of word insertions or additions the system makes when the image does not contain those specific words. The term N is the total number of correct words in the image that should have been extracted by the system if there were no previous faults. 

CER is the metric that measures the percentage of characters that are not or incorrectly recognized by the OCR system. In this context, characters are known as any additional aspects of a word that are necessary to its readability. Examples can be apostrophes or dashes. CER is calculated through the following formula, CER = (S + D + I) / N, which is similar to that of word error rate. In this formula, S is the number of character substitutions meaning any character that was incorrectly recognized by the OCR system. D is the number of character deletions and I is the number of character insertions or additions. Similarly to the WER formula, N is the total number of characters that should be correctly recognized by the OCR system if it ran efficiently.

For the specific use of our project, we used these two metrics to determine if the problem we were trying to solve was credible in the context of efficiency and correctness. We instituted the parameters of these metrics into the sample sizing of our data set and correlated the differences depending on the error rates of the sample.  We used the WER and CER metrics to quantify the model's accuracy in a context that was able to be used to predict the accuracy of this model in the future, and find places where suggestions can be made on what can be fixed. The Keras-OCR library provides functions that calculate these metrics within the data analyzation method of choice, which in our case was Jupyter Notebook and the Pandas library. Based on the predicted outputs and the actual extraction texts, we were able to understand how the inaccuracy of this model may affect its ability to create feasible environments that are meant to aid in accessibility efforts.  These metrics usually handle the correlation of the predicted and ground truth or actual text from the image and compute the error rates correctly, which can be used to modify the parameters of the extraction efforts itself. 

<img width="600" alt="Screen Shot 2023-04-28 at 1 44 09 PM" src="https://user-images.githubusercontent.com/51467244/235228360-afb00687-6a3f-4fad-8a19-aeae00b29df1.png">

## Conclusion

To determine the WER and CER of our model, we iterated through the Keras OCR pipeline results and the annotated images dataset to compare the actual and predicted results. We ended up with a Mean Word Error Rate of 67.42% and a Mean Character Error Rate of 27.36%. The Mean Word Error Rate indicates that almost 7 out of 10 words were wrongly transcribed. The Mean Character Error Rate indicates that nearly every third character was wrongly transcribed. Though these results imply poor OCR output performance, we hypothesize that this is due to us sampling a small portion of the image dataset.

```
# Importing jiwer 

from jiwer import wer
from jiwer import cer

actual = "scouts together again!"
predicted = "scouts tocther againt"

error_wer=0
error_cer=0

dfs = []


def compare(img_fn, kerasocr_df):
    img_id = img_fn.split('/')[-1].split('.')[0]
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    keras_results = kerasocr_df.query('img_id == @img_id')[['text','bbox']].values.tolist()
    keras_results = [(x[0], np.array(x[1])) for x in keras_results]


for img_fn in img_fns[:25]:
    error_wer= error_wer + wer(img_fn, kerasocr_df)
    error_cer= error_cer + cer(img_fn, kerasocr_df)
    

print(error_wer/25)
print(error_cer/25)

```

<img width="719" alt="Screen Shot 2023-05-01 at 7 56 04 PM" src="https://user-images.githubusercontent.com/51467244/235556970-3db0dbc5-1803-495c-8919-3fe25c5e5e6a.png">


## Reflections & Next Steps

**What we learned:**

In this project, we learned how to implement OCR models, focusing on Keras-OCR, as well as compute metrics specific to OCR, such as the Word Error Rate and Character Error Rate as created by Keras-OCR. Our model is currently performing with a Mean Word Error Rate of 67.42% and a Mean Character Error Rate of 27.36%. We also learned how to troubleshoot problems occurring from large datasets, specifically obtained through Kaggle downloads. We learned to sample down on the data, as well as scale up - depending on the circumstances and the power capabilities of our machines.

- How to work with large datasets: download and sample
- How to implement OCR models, particularly Keras-OCR
- How to compute OCR specific metrics (WER and CER)
- How to troubleshoot problems occurring from large datasets: sampling down and scaling up

**Challenges Faced:**

Along the way, we were presented with several challenges, some technical and some due to working with new materials. Our first issue came from the dataset on Kaggle being too large, creating an issue with downloading. Another problem occurred with regard to Google Colab, as we could not edit the file at the same time and kept deleting each other’s progress. To resolve this, we decided to have one person use Jupyter Notebook and work on it together through Zoom Meetings. Additionally, we worked with a module, Keras-OCR, that was new to all of us, and used CER and WER to evaluate the error rates instead of using accuracy, as we did in class. We iterated these formulas calculations through the Test annotations and the Keras OCR results, facing struggles with formatting.

- Working with a large dataset -> how to sample down and slowly increase to improve test results
- How to coordinate a coding project between team members. We initially tried using Google Colab, but found it difficult to use for our goals. 
- Worked with a new module (Keras-OCR)
- Evaluating models: CER and WER instead of accuracy
  - Had to iterate these formulas calculations through the Test annotations and the Keras OCR results (format difficulties)

**Next Steps:**

Moving further in this project, we will integrate a text-to speech dataset to further assist people with visual impairments and to ensure our findings are helpful. We will also work to improve the WER and CER of models, by lowering both error rates to improve the OCR performance. However, those metrics do not accurately reflect the overall quality of the OCR output, especially factors such as grammar or context of the text. Thus, we will continue to evaluate the OCR performance using human evaluations. Additionally, we want to evaluate the semantics of the word detection process. We will also work further to collect data on individuals with visual impairments to discover or create a more useful dataset for the task, as the one we were using in this project consisted of simple images and created more generic results.

- Integrate a text-to-speech dataset to complete the project
- Improve WER and CER of models by increasing the number of training images used 
- Conduct research on individuals with visual impairments to identify user needs and curate a dataset that would be more representative of actual usage cases


## References

- Mulla, R. (2022, July). TextOCR - Text Extraction from Images Dataset, Version 1. Retrieved March 21, 2023 from https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset. 

- Ch’ng, C, et al. (2022, April). Total-Text-Dataset, Version 23. Retrieved March 21, 2023 from https://github.com/cs-chan/Total-Text-Dataset.

- Davydova, O. (2017, September 26). 7 types of Artificial Neural Networks for Natural Language Processing. Medium. https://medium.com/@datamonsters/artificial-neural-networks-for-natural-language-processing-part-1-64ca9ebfa3b2. 

- Leung, K. (2021, June 24). Evaluate OCR Output Quality with Character Error Rate (CER) and Word Error Rate (WER). Towards Data Science. https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510.

- Vaessen, N. (2023, March 28). jiwer 3.0.1. pypi. Retrieved May 1, 2023, from https://pypi.org/project/jiwer/ 

- Parker, A. (2023, February 3). Optical Character Recognition: Then and Now. Weights & Biases. https://wandb.ai/andrea0/optical-char/reports/Optical-Character-Recognition-Then-and-Now--VmlldzoyMDY0Mzc0. 


