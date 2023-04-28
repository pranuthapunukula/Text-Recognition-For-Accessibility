# Text Recognition For Accessibility

The **goal** of our project is to promote accessibility for people with visual impairments by identifying text from a picture and reading it out. This project
is important because improving accessibility provides equal access to people with disabilities. This ensures that all people can perceive, understand,
and navigate the world equally and with assistive technology when needed.

The **dataset** we will be using is the **Kaggle Text Extraction from Images Dataset**. The dataset and more details can be found at: https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset?resource=download

Our **proposed method** for building a product is using a Neural Network to identify text from images using Natural Language Processing.
 

 
## Dataset

TextOCR requires models to perform text-recognition on arbitrary shaped scene-text present on natural images. 
This OCR provides ~1M high quality word annotations on TextVQA images allowing application of end-to-end reasoning on downstream tasks such as visual question answering or image captioning.

This data is available with a *CC0: Public Domain license*. 



## Plotting Example Images

```
# Plotting example images

img_fns = glob.glob('../Final_Project/train_val_images/train_images/*')
fig, axs = plt.subplots(5, 5, figsize=(20, 20))
axs = axs.flatten()
for i in range(25):
axs[i].imshow(plt.imread(img_fns[i]))
axs[i].axis('off')
image_id = img_fns[i].split('/')[-1].rstrip('.jpg')
n_annot = len(annot_df.query('image_id == @image_id'))
axs[i].set_title(f'{image_id} - {n_annot}')
plt.show()
```

<img width="650" alt="Screen Shot 2023-04-25 at 4 02 38 PM" src="https://user-images.githubusercontent.com/51467244/235227359-653ba86a-952e-481b-858a-a1aa5db9f6d7.png">


## Extracting Text from Images using Keras OCR

keras-ocr is an OCR library built on top of the popular deep learning framework, Keras. 
It utilizes the CRAFT: Character-Region Awareness For Text detection algorithm with a VGG model as the backbone.  
- It uses multiple machine algorithms for pattern recognition to determine the layout and presence of text image files
- It is trained to recognize characters, shapes, and numbers in order to recognize text in images 
- This is done using a combination of hardware, such as optical scanners and software capable of image processing
- It is trained on re-purposed data from scanned documents, camera images, and image-only pdfs

<img width="500" alt="Screen Shot 2023-04-28 at 2 06 17 PM" src="https://user-images.githubusercontent.com/51467244/235232311-6bdf6d2e-2870-4de7-98f6-f9757eaa7703.png">

*VGG illustration - U Toronto's Prof Davi Frossard*

Sources:
- https://pypi.org/project/keras-ocr/
- https://wandb.ai/andrea0/optical-char/reports/Optical-Character-Recognition-Then-and-Now--VmlldzoyMDY0Mzc0


```
import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()
results = pipeline.recognize([img_fns[1]])
keras_ocr.tools.drawAnnotations(plt.imread(img_fns[1]), results[0])
pd.DataFrame(results[0], columns=['text', 'bbox'])
fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(plt.imread(img_fns[1]), results[0], ax=ax)
ax.set_title('Keras OCR Result Example')
plt.show()

```
<img width="350" alt="Screen Shot 2023-04-28 at 1 40 05 PM" src="https://user-images.githubusercontent.com/51467244/235227638-1623994a-f0af-4a7d-a64f-7204f9d2be59.png">

```
results = pipeline.recognize([img_fns[1]])
pd.DataFrame(results[0], columns=['text', 'bbox'])
fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(plt.imread(img_fns[1]), results[0], ax=ax)
ax.set_title('Keras OCR Result Example')
plt.show()

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
```
<img width="500" alt="Screen Shot 2023-04-28 at 1 43 52 PM" src="https://user-images.githubusercontent.com/51467244/235228287-fbb6dccd-f1e0-42e3-9873-ddd804028902.png">

<img width="300" alt="Screen Shot 2023-04-25 at 4 29 20 PM" src="https://user-images.githubusercontent.com/51467244/235227852-025e3cf1-774f-43a8-8ff7-23fbeb83089c.png"> <img width="300" alt="Screen Shot 2023-04-25 at 4 29 14 PM" src="https://user-images.githubusercontent.com/51467244/235227864-9a412ac0-1fd3-4eef-bb8b-dd5f1f4133ee.png">




## Generating Metrics - How well does our model perform?

Initially planned to use accuracy to determine model performance, but opted to use Word Error Rate (WER) and Character Error Rate (CER) instead. This is a better objective metric because it integrates potential substitution, deletion, and insertion errors. This offers us a benchmark to iterate our model to steadily improve performance metrics. 

```
#Adding Code Here
```

<img width="500" alt="Screen Shot 2023-04-28 at 1 44 09 PM" src="https://user-images.githubusercontent.com/51467244/235228360-afb00687-6a3f-4fad-8a19-aeae00b29df1.png">

Source: https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510


## Reflections & Next Steps

What we learned:
- How to work with large datasets: download and sample
- How to implement OCR models, particularly Keras-OCR
- How to compute OCR specific metrics (WER and CER)
- How to troubleshoot problems occurring from large datasets: sampling down and scaling up

Challenges Faced:

- Working with a large dataset -> how to sample down and slowly increase to improve test results
- How to coordinate a coding project between team members. We initially tried using Google Colab, but found it difficult to use for our goals. 
- Worked with a new module (Keras-OCR)
- Evaluating models: CER and WER instead of accuracy
  - Had to iterate these formulas calculations through the Test annotations and the Keras OCR results (format difficulties)

Next Steps:

- Integrate a text-to-speech dataset to complete the project
- Improve WER and CER of models by increasing the number of training images used 
- Conduct research on individuals with visual impairments to identify user needs and curate a dataset that would be more representative of actual usage cases





