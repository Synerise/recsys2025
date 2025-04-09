# Universal Behavioral Modeling Data Challenge

## Introduction

### Why a Universal Behavioral Modeling Challenge?

The challenge is designed to promote a unified approach to behavior modeling. Many modern enterprises rely on machine learning and predictive analytics for improved business decisions. Common predictive tasks in such organizations include recommendation, propensity prediction, churn prediction, user lifetime value prediction, and many others. A central piece of information used for these predictive tasks are logs of past behavior of users e.g., what they bought, what they added to their shopping cart, which pages they visited. Rather than treating these tasks as separate problems, we propose a unified modeling approach.

To achieve this, we introduce the concept of Universal Behavioral Profiles — user representations that encode essential aspects of each individual’s past interactions. These profiles are designed to be universally applicable across multiple predictive tasks, such as churn prediction and product recommendations. By developing representations that capture fundamental patterns in user behavior, we enable models to generalize effectively across different applications.

### Challenge Overview

The objective of this challenge is to develop Universal Behavioral Profiles based on the provided data, which includes various types of events such as purchases, add to cart, remove from cart, page visit, and search query. These user representations will be evaluated based on their ability to generalize across a range of predictive tasks. The task of the challenge participants is to submit user representations, which will serve as inputs to a simple neural network architecture. Based on the submitted representations, models will be trained on several tasks, including some that are disclosed to participants, called "open tasks," as well as additional hidden tasks, which will be revealed after the competition ends. The final performance score will be an aggregate of results from all tasks. We iterate model training and evaluation automatically upon submission. The only task for participants is to submit universal user representations.

- Participants are asked to provide user representations — Universal Behavioral Profiles
- Downstream task training is conducted by the organizers, however, the competition pipeline is publicly available and is presented in this repository
- Model for each downstream task is trained separately, but using the same embeddings (user representations)
- Performance will be evaluated based on all downstream tasks

### Open Tasks
- **Churn Prediction:** Binary classification into 1: user will churn or 0: user will not churn. Churn task is performed on a subset of active users with at least one `product_buy` event in history (data available for the participants)
Task name: `churn`
- **Categories Propensity:** Multi-label classification into one of 100 possible labels. The labels represent the 100 most often purchase product categories.
Task name: `propensity_category`
- **Product Propensity:** Multi-label classification into one of 100 possible labels. The labels represent the 100 most often purchase products in train target data.
Task name: `propensity_sku`

### Hidden Tasks
In addition to the open tasks, the challenge includes hidden tasks, which remain undisclosed during the competition. The purpose of these tasks is to ensure that submitted Universal Behavioral Profiles are capable of generalization rather than being fine-tuned for specific known objectives. Similar to the open tasks, the hidden tasks focus on predicting user behavior based on the submitted representations, but they introduce new contexts that participants are not explicitly optimizing for.

After the competition concludes, the hidden tasks will be disclosed along with the corresponding code, allowing participants to replicate results.

## Dataset

We release an anonymized dataset containing real-world user interaction logs.
Additionally, we provide product properties that can be joined with `product_buy`, `add_to_cart`, and `remove_from_cart` event types.
Each source has been stored in a separate file.

**Note**  
All recorded interactions can be utilized to create Universal Behavioral Profiles; however, participants are required to submit behavioral profiles for only a subset of 1,000,000 users, which will be used for model training and evaluation.


|          |    product_buy    |    add_to_cart    |    remove_from_cart    |      page_visit  |      search_query  |
|:---------|:-----------------:|:-----------------:|:----------------------:|:----------------:|:------------------:|
|   Events |     1,682,296     |     5,235,882     |     1,697,891          |     150,713,186  |     9,571,258      |

### Dataset Description

#### Columns

**product_properties**:
- **sku (int64):** Numeric ID of the item.
- **category (int64):** Numeric ID of the item category.
- **price (int64):** Numeric ID of the bucket of item price (see section [Column Encoding](https://github.com/Synerise/recsys2025#column-encoding)).
- **name (object):** Vector of numeric IDs representing a quantized embedding of the item name (see section [Column Encoding](https://github.com/Synerise/recsys2025#column-encoding)).

**product_buy**:
- **client_id (int64):** Numeric ID of the client (user).
- **timestamp (object):** Date of event in the format YYYY-MM-DD HH:mm:ss.
- **sku (int64):** Numeric ID of the item.

**add_to_cart**:
- **client_id (int64):** Numeric ID of the client (user).
- **timestamp (object):** Date of event in the format YYYY-MM-DD HH:mm:ss.
- **sku (int64):** Numeric ID of the item.

**remove_from_cart**:
- **client_id (int64):** Numeric ID of the client (user).
- **timestamp (object):** Date of event in the format YYYY-MM-DD HH:mm:ss.
- **sku (int64):** Numeric ID of the item.

**page_visit**:
- **client_id (int64):** Numeric ID of the client.
- **timestamp (object):** Date of event in the format YYYY-MM-DD HH:mm:ss.
- **url (int64):** Numeric ID of visited URL. The explicit information about what (e.g., which item) is presented on a particular page is not provided.

**search_query**:
- **client_id (int64):** Numeric ID of the client.
- **timestamp (object):** Date of event in the format YYYY-MM-DD HH:mm:ss.
- **query (object):** Vector of numeric IDs representing a quantized embedding of the search query term (see section [Column Encoding](https://github.com/Synerise/recsys2025#column-encoding)).

#### Column Encoding

**Text Columns ('name', 'query')**:  
In order to anonymize the data, we first embed the texts with an appropriate large language model (LLM). Then, we quantize the embedding with a high-quality embedding quantization method. The final quantized embedding has the length of 16 numbers (buckets) and in each bucket, there are 256 possible values: {0, …, 255}.

**Decimal Columns ('price')**:  
These were originally floating-point numbers, which were split into 100 quantile-based buckets.

## Data Format

This section describes the format of the competition data.
We provide a data directory containing event files and two subdirectories: `input` and `target`.

**Note**  
For the purpose of running training and baseline code from this repository, it is important to keep the data directory structure intact.

### 1. Event and properties files
The event data, which should be used to generate user representations, is divided into five Parquet files. Each file corresponds to a different type of user interaction available in the dataset (see section [Dataset Description](https://github.com/Synerise/recsys2025#dataset-description)):

- **product_buy.parquet**
- **add_to_cart.parquet**
- **remove_from_cart.parquet**
- **page_visit.parquet**
- **search_query.parquet**

Product properties are stored in:

- **product_properties.parquet**

### 2. `input` directory
This directory stores a NumPy file containing a subset of 1,000,000 `client_id`s for which Universal Behavioral Profiles should be generated:

- **relevant_clients.npy**

Using the event files, participants are required to create Universal Behavioral Profiles for the clients listed in `relevant_clients.npy`. These clients are identified by the `client_id` column in the event data.

The generated profiles must follow the format outlined in the **Competition Entry Format** section and will serve as input for training models across all specified tasks, including churn prediction, product propensity, category propensity, and additional hidden tasks. The code for the downstream tasks is fixed and provided to participants (see [Model Training](https://github.com/Synerise/recsys2025#model-training) section).

### 3. `target` directory
This directory stores the labels for propensity tasks. For each propensity task, target category names are stored in NumPy files:

- **propensity_category.npy**: Contains a subset of 100 categories for which the model is asked to provide predictions
- **popularity_propensity_category.npy**: Contains popularity scores for categories from the `propensity_category.npy` file. Scores are used to compute the Novelty measure. For details, see the [Evaluation](https://github.com/Synerise/recsys2025#evaluation) section
- **propensity_sku.npy**: Contains a subset of 100 products for which the model is asked to provide predictions
- **popularity_propensity_sku.npy**: Contains popularity scores for products from the `propensity_sku.npy` file. These scores are used to compute the Novelty measure. For details, see the [Evaluation](https://github.com/Synerise/recsys2025#evaluation) section
- **active_clients.npy**: Contains a subset of relevant clients with at least one `product_buy` event in history (data available for the participants). Active clients are used to compute churn target. For details, see the [Open Tasks](https://github.com/Synerise/recsys2025#open-tasks) section

These files are specifically used to create the ground truth labels for the propensity tasks. The target (ground truth) for each task is automatically computed by the `TargetCalculator` class in `universal_behavioral_modeling_challenge.training_pipeline.target_calculators`.

**Note**  
To run internal experiments with this repository, the event data should be split into `input` and `target` chunks, which are stored in the `input` and `target` directories, respectively. This setup imitates the official evaluation pipeline; however, the official train and validation target data are not provided to competitors. To create the data split, see the [Data Splitting](https://github.com/Synerise/recsys2025#data-splitting) section.

## Competition Entry Format

**Participants are asked to prepare Universal Behavioral Profiles — user representations that will serve as input to the first layer of a neural network with a fixed, simple architecture.** For each submission, the models will be trained and evaluated by the organizers, and the evaluation outcome will be displayed on the leaderboard. However, we make the training pipeline and model architecture available for participants to use in internal testing.

Each competition entry consists of two files: `client_ids.npy` and `embeddings.npy`

#### `client_ids.npy`
   - A file that stores the IDs of clients for whom Universal Behavioral Profiles were created
   - `client_ids` must be stored in a one-dimensional NumPy ndarray with `dtype=int64`
   - The file should contain client IDs from the `relevant_clients.npy` file, and the order of IDs must match the order of embeddings in the embeddings file

#### `embeddings.npy`
   - A file that stores Universal Behavioral Profiles as a **dense user embeddings matrix**
   - Each embedding corresponds to the client ID from `client_ids.npy` with the same index
   - Dense embeddings must be stored in a two-dimensional NumPy ndarray, where:
     - The first dimension corresponds to the number of users and matches the dimension of the `client_ids` array
     - The second dimension represents the embedding size
     - The `dtype` of the embeddings array must be `float16`
   - The embedding size cannot exceed `max_embedding_dim = 2048`

**It is crucial that the order of IDs in the `client_ids` file matches the order of embeddings in the embeddings file to ensure proper alignment and data integrity.**

### **IMPORTANT! The maximum length of the embedding vectors is 2048.**

## Competition Entry Validator

Competitors must ensure that their submitted files adhere to this structure for successful participation in the competition. The entry format can be validated using the provided validation script:

`universal_behavioral_modeling_challenge/validator/run.py`

### **Arguments**
   - `--data-dir`: The directory containing the data provided by the organizer, including `relevant_clients.npy` file in `input` subdirectory (described in **Data Format** section)
   - `--embeddings-dir`: The directory where the `client_ids` and `embeddings` files are stored

### **Running the Validator**

```bash
python -m validator.run --data-dir <data_dir> --embeddings-dir <your_embeddings_dir>
```

## Model training
Model training is conducted by challenge organizers. Multiple indepedent models with identical architecutre are trained for downstream tasks (churn, propensity, and hidden tasks) and the combined score is presented on the leaderboard. The training process is fixed for every task and all competition entries. Our objective is to evaluate the expressive power of created Universal Behaviorl Profiles, not the model architecture itself.

### Model architecture
- the model architecture consists of three Inverted Bottleneck blocks, adapted from *Scaling MLPs: A Tale of Inductive Bias* (https://arxiv.org/pdf/2306.13575.pdf), with layer normalization and residual connections; see `UniversalModel` class in `universal_behavioral_modeling_challenge.training_pipeline.model`
- the input to the first layer of the network are embeddings provided in the competition entry
- the output is task-specific and computed by the `TargetCalculator` class in `universal_behavioral_modeling_challenge.training_pipeline.target_calculators`
- model hyperparameters are fixed and the same for each task and each competition entry:
   - BATCH_SIZE = 128
   - HIDDEN_SIZE_THIN = 2048
   - HIDDEN_SIZE_WIDE = 4096
   - LEARNING_RATE = 0.001
   - MAX_EMBEDDING_DIM = 2048
   - MAX_EPOCH = 3

**Note**  
For model evaluation, we consider the best score out of 3 epochs.

## Evaluation

The primary metric used to measure model performance is AUROC. We use `torchmetrics` implementation of AUROC. Additionally, the performance of Category Propensity and Product Propensity models is evaluated based on the novelty and diversity of the predictions. In these cases, the task score is calculated as a weighted sum of all metrics:
```
0.8 × AUROC + 0.1 × Novelty + 0.1 × Diversity
```
### Diversity

To compute the diversity of single prediction, we first apply element-wise sigmoid to the predictions, and l1 normalize the result. The diversity of the prediction is the entropy of this distribution.

The final diversity score is computed as the average diversity of the model's predictions.

### Novelty

The popularity of a single prediction is the weighted sum of the popularities of the top `k` recommended targets in the prediction. This is normalized so that a popularity score of `1` corresponds to the following scenario:
> The model's top `k` predictions are the `k` most popular items, and the model is absolutely certain about predicting all of these items.

The popularity score is then computed as the average popularity of the model's predictions. Finally, we compute the novelty of the predictions as `1 - popularity`.
 
Due to the sparsity of the data, the popularity scores, as computed so far are close to 0, and thus the corresponding raw novelty scores are really close to 1. To make the measure more sensitive to small changes near 1, we raise the raw popularity score to the 100th power.

### Final leaderboard score

For each task, a leaderboard is created based on the respective task scores. The final score, which evaluates the overall quality of user representations and their ability to generalize, is determined by aggregating ranks from all per-task leaderboards using the Borda count method. In this approach, each model's rank in a task leaderboard is converted into points, where a model ranked `k`-th among `N` participants receives `N - k` points. The final ranking is based on the total points accumulated across all tasks, ensuring that models performing well consistently across multiple tasks achieve a higher overall score.

## Competition submission

1. Organizers provide the input set of event data as described in the [Data Format](https://github.com/Synerise/recsys2025#data-format) section
2. Competitors are asked to create user embeddings (Universal Behavioral Profiles) based on the provided data
3. Created embeddings are submitted following the **Competition Entry Format**
4. Organizers are using submitted embeddings to train models in multiple **Downstream Tasks**
5. Models are validated on the left-out subset of data
6. Validation results are presented on the leaderboards


## Competition code
We provide a framework that participants can use to test their solutions. The same code is used in the competition to train models for downstream tasks. Only targets for hidden tasks are not included in the provided code.

### Requirements
Requirements are provided in the `requirements.txt` file.

### Data Splitting
Running the competition code for internal tests requires splitting raw event data into three distinct time windows: input events, events to compute train target, and events to compute validation target.
The first set of events is training input data that are used to create users' representations. These representations serve as an input to train downstream models. For baseline solutions with user representation methods see `baseline` module in this repository.
The training target is not included in data tables explicitly, but is computed on the fly based on events in the training target time window. It consists of 14 days after the last timestamp in the input data. The model is trained to predict events in the target based on input user representations.

**Target example:**
To create a propensity target, we check if the user made any purchase in a given category within the provided target time window. Propensity target categories are provided in separate files: `propensity_category.npy` and `propensity_sku.npy`. In the case of a churn target, we check if the user made any purchase in the provided target data sample.

The next 14 days after the training target set are used to compute the validation target and measure model performance.

**IMPORTANT! This data-splitting procedure is meant for internal testing. In the official competition settings, users' representations — which are the competition entry — should be created based on ALL provided events. The official training and validation targets are hidden from the competitors.**

### Split data script
We provide a script to split data according to the described procedure:
`data_utils/split_data.py`

**Arguments**

- `--challenge-data-dir`: Competition data directory which should consist of event files, product properties file and two subdirectories — input (with `relevant_clients.npy`) and target (with `propensity_category.npy` and `propensity_sku.npy`).

**Output**
Input events are saved as Parquet files in the input subdirectory in `--challenge-data-dir`. Train and validation target events are saved as Parquet files in the target subdirectory in `--challenge-data-dir`

**Running**
Run the script:
```bash
python -m data_utils.split_data --challenge-data-dir <your_challenge_data_dir>
```

### Model training script
**Arguments**
- `--data-dir`: Directory where competition target and input data are stored.
- `--embeddings-dir`: Directory where Universal Behavioral Profiles, which are used as model input embeddings are stored. Embeddings should be stored in the format described in the **Competition Entry Format** section.
- `--tasks`: List of tasks to evaluate the model on, possible values are: `churn`, `propensity_category`, `propensity_sku`.
- `--log-name`: Name for the experiment, used for logging.
- `--num-workers`: Number of subprocesses for data loading.
- `--accelerator`: Type of accelerator to use. Argument is directly passed to `pl.LightningModule`. Possible values include: `gpu`, `cpu`. For more options, [see here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator) .
- `--devices`: List of devices to use for training. Note that using `auto` when `accelerator="gpu"` sometimes produces undesired behavior, and may result in slower training time.
- `--neptune-api-token` (optional): API token for Neptune logger. If not specified, the results are logged offline.
- `--neptune-project` (optional): Name of Neptune project in the format `<workspace>/<project>` to log the results of the experiment to. If not specified, the results are logged offline.
- `--disable-relevant-clients-check` (optional): This flag disables the validation check that ensures the `client_ids.npy` file from the submission matches the contents of `relevant_clients.npy`. It allows training to be run on a different set of clients than the relevant clients.  

**Note**
For the official submission, the relevant clients validation check will be enabled, and embeddings must be provided **for all and only the relevant clients**. However, the --disable-relevant-clients-check flag should be used for internal experiments, as not all relevant clients remain in the input data after using the `data_utils.split_data` script.

**Running scripts**

Offline logging:
```bash
python -m training_pipeline.train --data-dir <your_splitted_challenge_data_dir> --embeddings-dir <your-embeddings-dir> --tasks churn propensity_category propensity_sku --log-name <my_experiment> --accelerator gpu --devices 0 --disable-relevant-clients-check
```
Logging into a Neptune workspace:
```bash
python -m training_pipeline.train --data-dir <your_splitted_challenge_data_dir> --embeddings-dir <your-embeddings-dir> --tasks churn propensity_category propensity_sku --log-name <my_experiment> --accelerator gpu --devices 0 --neptune-api-token <your-api-token> --neptune-project <your-worskspace>/<your-project> --disable-relevant-clients-check
```

## Baselines
In the `baseline` directory, we provide scripts with an example competition entry and an additional README_AGGREGATED_FEATURES.md file with more detailed instructions.  

---
*In case of any problems, you can contact the organizers by email via the address [recsyschallenge\@synerise.com](mailto:recsyschallenge\@synerise.com).*