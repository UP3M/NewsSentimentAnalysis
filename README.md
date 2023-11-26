# News Sentiment Analysis for Company


## Project Description 

The project is about application of the NLP model on sentiment analysis of news for specific companies. The  background of the project is there are many chances for general market analysis based on sentiment analysis of news, but few for learning about or forecasting the performance of a given company based on pertinent headlines. The existing approach uses sentiment analysis only on the news that explicitly mentions the company on the headline. In fact, some news which does not explicitly state the name of the company might have potential significant influence. 

The project's focus on sentiment analysis specifically tailored to individual companies addresses a critical gap in existing market analysis tools. While general sentiment analysis provides valuable insights into market trends, understanding how news affects the perception and performance of a particular company adds a layer of specificity that is invaluable for investors seeking targeted and informed decision-making.

Moreover, the adoption of two distinct NLP models, namely BERT and the combination of FinBERT with Cosine Similarity, showcases the project's commitment to robust analysis. By comparing the performance of these models, the project not only contributes to the growing body of literature on sentiment analysis but also provides practitioners with insights into the strengths and weaknesses of different methodologies.

In conclusion, the significance of the News Sentiment Analysis for Company project lies in its innovative approach to understanding the intricate relationship between news sentiment and company performance. By pushing the boundaries of traditional analysis methods and leveraging advanced NLP techniques, this project has the potential to reshape how market participants perceive and respond to news events, ultimately contributing to more informed decision-making and a deeper understanding of the dynamic interplay between information, sentiment, and financial markets.

## How To Run:
#### 1. The dataset is available in "dataset" folder.
#### 2. The pretrained model for 1D-CNN and fine-tuned model for BERT are available on [https://drive.google.com/file/d/1D2Dwf15oGeeeQgGcnZTnJ8baNHCILR75/view?usp=sharing](https://drive.google.com/drive/folders/1HeM7KUTfEvprYYBKBWgK2t-SBNaabyZc?usp=sharing).
#### 3. We visualize our project using streamlit (all necessary file in "visualization" folder), to clone it and make it work, simply do following step:
 3.1. Download all pretrained model and change the path inside app.py
 
 3.2. Install all necessary library by using pip command (all required library is writen in requirement.txt)
 
 3.3. Run streamlit using command "streamlit run app.py"
 
 ##### N.B.: recommended to use virtual environment
