# Integrating Knowledge Graph Information into BERT Embeddings for Enhanced Entity Classification

Beyza İrem Urhan, Department of Computer Engineering, Boğaziçi University

*Final report of the CMPE 58T (Advanced Natural Language Processing) application project, Spring 2024. Reproduced as submitted, apart from formatting.*

## Abstract

This paper presents a method to enhance BERT embeddings by integrating them with knowledge graph (KG) embeddings for improved performance in entity classification tasks. BERT embeddings capture rich semantic information from text, while KG embeddings provide structured knowledge about entities and their relationships. We explore two approaches for this integration and evaluate their performance using the Re-TACRED dataset. Despite some challenges, the combined embeddings may show potential for improved semantic understanding and classification accuracy.

## 1 Introduction

Word embeddings have revolutionized natural language processing (NLP), enabling words to be represented as vectors that capture both semantic and syntactic relationships. This has facilitated numerous applications such as sentiment analysis, machine translation, and information retrieval. BERT (Bidirectional Encoder Representations from Transformers) is particularly notable for its context-aware embeddings that have improved performance across various NLP tasks. However, BERT embeddings might lack the explicit relational information between entities that a knowledge graph (KG) can provide. This project aims to enhance BERT embeddings by integrating them with KG embeddings to improve semantic understanding and accuracy in entity classification tasks.

## 2 System Description

This section describes two approaches to integrate BERT and KG embeddings to improve the performance of entity classification tasks.

### 2.1 BERT Embeddings

BERT embeddings are derived from a transformer model pre-trained on large text corpora. They capture context-dependent semantics, which are crucial for various NLP tasks. BERT has set new benchmarks in tasks like question answering, named entity recognition (NER), and sentence classification.

### 2.2 Knowledge Graph Embeddings

Knowledge graphs (KGs) store structured information about entities and their interrelations. KG embeddings, generated using models like TransE, embed this structured information into continuous vector spaces. These embeddings are useful for tasks such as link prediction and entity classification.

### 2.3 Integrating BERT and KG Embeddings

Integrating BERT and KG embeddings aims to combine the strengths of both: the contextual understanding from BERT and the structured relational knowledge from KGs. Previous work has explored similar integrations to enhance language models with external knowledge, improving their performance in knowledge-intensive tasks.

## 3 Experimental Setup

### 3.1 Datasets

**Re-TACRED.** Re-TACRED is a dataset for relation extraction, providing annotated relationships between entities in sentences. It includes a wide range of entity types and relations, making it suitable for training and evaluating our models.

**Knowledge graph datasets.**

- DBpedia50: A subset of DBpedia containing structured information about entities.
- WN18RR: A subset of WordNet, focused on lexical relations.
- FB15k-237: A subset of Freebase, focusing on various entity relations.

### 3.2 Embedding Models

**BERT embeddings.** We use the `bert-base-uncased` model from Hugging Face's Transformers library. The embeddings have a dimension of 768.

**TransE model for KG embeddings.** The TransE model is used to generate KG embeddings. The embedding dimension is set to 50. The TransE model learns to embed entities and relationships by minimizing a margin-based ranking loss function.

### 3.3 Integration Approaches

**BERT-only embeddings.** In this baseline approach, we use BERT embeddings directly for the entity classification task. The BERT model is fine-tuned on the Re-TACRED dataset.

**Combined BERT and KG embeddings.** In this approach, we concatenate BERT embeddings with KG embeddings. The combined embeddings are then used for the entity classification task.

### 3.4 Experimental Setup

- Training: Both BERT-only and combined embeddings are fine-tuned on the Re-TACRED dataset.
- Optimization: We use the Adam optimizer with a learning rate of 2e-5 for BERT and 1e-3 for the TransE model.
- Evaluation: Models are evaluated using precision, recall, and F1-score on the test set of Re-TACRED.

## 4 Results and Discussion

### Challenges in Entity Mapping

One of the significant challenges encountered during this project was the difficulty in mapping entities between the different datasets, specifically the Re-TACRED dataset and the knowledge graph datasets (DBpedia50, WN18RR, and FB15k-237). The Re-TACRED dataset consists of entity names and relations that are often not directly aligned with the entities and relations found in the knowledge graphs used for generating KG embeddings. This misalignment led to issues where many entities from the Re-TACRED dataset could not be found in the knowledge graphs, resulting in missing or zero vectors when attempting to retrieve KG embeddings.

For instance, the entity "New York Immigration Coalition" did not have a corresponding entry in the DBpedia50, WN18RR, or FB15k-237 knowledge graphs. This lack of correspondence was reflected in the model's output, where it repeatedly returned tensors filled with zeros for the KG embeddings.

Consequently, the integration of these embeddings with the BERT embeddings resulted in combined vectors that did not effectively leverage the KG information, as the KG component was essentially non-informative.

### Embedding Dimension Mismatch

Another issue faced was the embedding dimension mismatch between BERT embeddings and KG embeddings. BERT embeddings have a dimension of 768, while the KG embeddings generated by the TransE model had a dimension of 50. This disparity meant that when concatenating the embeddings to form a combined vector, the resultant vectors had a dimension of 818. This concatenation process, while straightforward, did not necessarily ensure that the combined embeddings were optimal for the downstream task of entity classification.

The process of combining these embeddings needs more sophisticated methods than simple concatenation to truly integrate the semantic information from BERT and the relational information from the KG embeddings effectively. Techniques such as attention mechanisms or more complex fusion methods might be necessary to achieve a seamless integration that can leverage the strengths of both types of embeddings.

### Model Training and Fine-Tuning

Training the modified BERT model, which included both the original BERT embeddings and the concatenated KG embeddings, presented additional challenges. The training process required significant computational resources, and the results did not show the expected improvements due to the issues mentioned above. The fine-tuning process, although configured correctly with appropriate hyperparameters (e.g., learning rate, batch size, and number of epochs), did not converge to a model that outperformed the BERT-only baseline.

The fine-tuning difficulties were further compounded by the presence of many zero vectors in the KG embeddings, which likely contributed to the model's inability to learn effectively from the combined embeddings. The presence of these zero vectors indicated a lack of useful KG information for many entities, thereby limiting the potential benefits of the KG-enhanced embeddings.

### Evaluation Metrics and Results

Due to the issues with entity mapping and embedding integration, we were unable to obtain meaningful evaluation results for the combined embeddings. The performance metrics, such as precision, recall, F1-score, and accuracy, were not significantly different from the baseline BERT-only model. This lack of improvement highlighted the necessity for more robust methods to handle the integration of BERT and KG embeddings and better strategies for entity alignment between disparate datasets.

### Future Directions for Improvement

- **Advanced entity linking techniques:** implementing more sophisticated entity linking methods to ensure better alignment between entities in the Re-TACRED dataset and the knowledge graph datasets.
- **Embedding fusion methods:** exploring advanced techniques such as attention mechanisms or other neural network architectures to better fuse BERT and KG embeddings.
- **Augmented knowledge graphs:** expanding the knowledge graphs to include more entities and relations relevant to the Re-TACRED dataset to reduce the number of zero vectors.
- **Context-aware KG embeddings:** developing context-aware KG embeddings that can dynamically adjust based on the textual context provided by BERT embeddings.

## 5 Conclusion

This study aimed to enhance BERT embeddings by integrating them with knowledge graph (KG) embeddings to improve entity classification tasks. While the idea of leveraging the structured knowledge from KGs in conjunction with the semantic richness of BERT embeddings holds promise, our implementation faced several challenges that hindered its success.

The primary challenge was the difficulty in mapping entities between the Re-TACRED dataset and the knowledge graph datasets (DBpedia50, WN18RR, and FB15k-237). This misalignment resulted in many entities lacking corresponding KG embeddings, which in turn led to the generation of non-informative zero vectors. Additionally, the simple concatenation of BERT and KG embeddings did not yield the expected improvements, indicating a need for more sophisticated embedding fusion methods.

Despite these challenges, the concept of integrating KG information into BERT embeddings remains a valuable avenue for future research. Improving entity linking techniques and exploring advanced methods for embedding fusion are critical next steps. Furthermore, expanding and enriching the knowledge graphs to better cover the entities in the target dataset will be essential for realizing the full potential of this integration.

In conclusion, while our current implementation did not achieve the desired performance improvements, it provided valuable insights into the challenges and complexities involved in combining BERT and KG embeddings. Addressing these challenges in future work could lead to significant advancements in enhancing semantic understanding and classification accuracy in NLP tasks.

## References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.
- Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. NeurIPS 26.
- Zhang, Y., Zhong, V., Chen, D., Angeli, G., & Manning, C. D. (2017). Position-aware Attention and Supervised Data Improve Slot Filling. EMNLP.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. EMNLP, 1532–1543.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781.
- Tokuhisa, R., Kawano, K., Nakamura, A., & Koide, S. (2022). Enhancing Contextual Word Representations Using Embedding of Neighboring Entities in Knowledge Graphs. COLING, 3175–3186.
- Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X. (2024). Unifying Large Language Models and Knowledge Graphs: A Roadmap. IEEE TKDE.
