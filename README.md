# Generating-Abstracts-of-research-papers-arXiv-using-pre-trained-transformers.
This is the NLP Project 
////////////////////////////////////////
Introduction
This project explores the task of generating abstracts for research papers using pre-trained transformers. The goal is to generate informative and concise summaries of research papers that can be used by researchers and academics to quickly identify relevant information and gain a deeper understanding of a paper's contribution.
////////////////////////////////////////
#  Generating Abstract of research papers(arXiv) using pretrained transformer (T5-Small)

## About the project
This projects develops a text summarization model that can accurately summarize research documents.
by Using a pretrained transformer (T5) to perform text summarization. It uses scientific_papers/arxiv dataset hosted on hugging face datasets. Project finetunes the pretrained T5-small model and reports its performance on test data comapred to untuned pretrained model.
## Intructions/Code Details
- requirements.txt <br>
	This captures the libraries needs to be installed to run this project
- scientificpapers-arxiv-summarization-t5mini.ipynb <br>
	To finetune the model on arxiv datat set run this notebook. This will also download the dataset from hugging face library

- arxiv-test-metric-v2.ipynb <br>
	To compute the ROUGE scores on test data using pretrained model run this notebook

- arxiv-test-metric-finetuned-v2.ipynb <br>
	To compute the ROUGE scores on test data using pretrained model run this notebook.Makesure to set the parameter 'modelpath' location of finetuned model

- Demo_Gradio_T5-small_finetuned.ipynb <br>
	To make a demo application using a gradio library run this notebook. Makesure to set the variable 'model_path' to point to the location of finetuned model

- sampletestdata.csv <br>
	This csv file holds few sample records taken from the scientific_papers/arxiv test dataset. that will be used in the gradio demo app

## Result
Rogue metric is chosen as the performance metric 

|Model   | rouge1  |  rouge2 | rougleL  |
|----------------------|---------|---------|----------|
| Pretrained T5-Small  |  26.71 |  6.15 | 16.48  |
| Fine Tuned T5-Small  |  **36.92** |  **11.69** | **11.69** |

## Citations
- Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li and Liu (2020) (T5): Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer [https://arxiv.org/abs/1910.10683]
- Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, Nazli Goharian (2018): A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents[https://arxiv.org/abs/1804.05685]
- Dataset (arxiv) hosted on Hugging Face[https://huggingface.co/datasets/ccdv/arxiv-summarization]
- T5 in different sizes[https://huggingface.co/docs/transformers/model_doc/t5]
- [https://kavita-ganesan.com/what-is-rouge-and-how-it-works-for-evaluation-of-summaries/]
- [https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb]
- [https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb]





///////////////////////////////////////
Conclusion
This user documentation manual provides an overview of the project and instructions for installing and using the code. We hope that this project will be useful for researchers and academics who want to quickly summarize research papers and gain a deeper understanding of their contributions.

If you have any questions or feedback, please feel free to contact us through the project repository.
