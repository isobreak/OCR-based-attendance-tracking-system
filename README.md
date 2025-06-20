# Attendance-tracking-using-OCR

The repository contains the code of the OCR-based attendance tracking system used in MEPhI. Based on the image and the list of possible student names, the system is able to recognize the list of those present. Model's weights could be found [here](https://drive.google.com/file/d/12sn9Mcr2HkITKLsXlHtjsE12lm7LhIxf/view?usp=sharing). Some of the examples are available in [data/results](data/results).

## System's Purpose
The use of sheets for self-recording of students present at lectures is a common practice at MEPhI. To automate the process of entering attendance data into the database, this service is used, which provides the opportunity to extract information about students present based on a worksheet.

## Result Examples
Examples of image processing are shown here. The system is able to match the text recognized in several images to one list of acceptable names.
<img src="data\results\match\626b34b2-e69c-4fa2-9e9d-94feb7c7794d.jpg" alt="Example of match result" width="100%"><br>
<img src="data\results\match\fe11508d-890d-4293-b051-68571d3a46fa.jpg" alt="Example of match result" width="100%"><br>
&emsp; **Note:** A confidence score is displayed to assist in the detection of possible errors. Areas are highlighted in red when the confidence score is low (<= 1).

## Solution Design
### Image Processing Pipeline
The recognition pipeline is implemented in the [processing.py](src/app/processing.py), and it consists of 4 stages:

1. **Word Detection**
   - There are 3 detection approaches implemented, which are based on:
     - [**DBSCAN**](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html): Clusters contour pixels based on their location - each cluster is considered a detected object. Is sensitive to the quality of image binarization. Сan be used on images with low text density.
     - [**Morphological Transformations**](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html): The fastest and least accurate of the methods. It is sensitive to the quality of image binarization. It can be used on images with low text density and strictly horizontal text arrangement.
     - [**Faster R-CNN**](https://arxiv.org/pdf/1506.01497): Has the best accuracy on complex data among the algorithms used, and the lowest speed. It is used in the final Pipeline class (processing.py).

2. **Word Clustering**
   - Clustering is carried out in order to separate all identified objects into groups containing information about one student. Assuming that students sign their names in separate lines, clustering is performed on the average Y value using [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

3. **Word Recognition**
   - [**TrOCR**](https://huggingface.co/raxtemur/trocr-base-ru) is used to recognize text within each bounding box.

4. **Names Matching**
   - The comparison is performed using [Levenshtein distances](https://ru.wikipedia.org/wiki/Расстояние_Левенштейна), normalized by the length of the corresponding words. The full algorithm is implemented in [src/app/processing.py](src/app/processing.py)


### Backend Application

*   **[FastAPI](https://fastapi.tiangolo.com/):** High-performance Python API framework.
*   **[RabbitMQ](https://www.rabbitmq.com/documentation.html):** Message broker for asynchronous communication.
*   **[Celery](http://docs.celeryq.dev/en/stable/):** Distributed task queue for background processing.
*   **[Docker](https://docs.docker.com/):** Containerization for consistent deployments.

## Project Structure

The project is organized into the following directories:

-   `data/`
    -   `results/`: &emsp;&emsp;&emsp;&emsp;&emsp;*image examples of each pipeline's stage*
        -   `clustering/`
        -   `detection/`
        -   `match/`
        -   `recognition/`
    -   `models/`: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;*missing from the repository*
        -   `production/`: &emsp;*the directory with weights could be taken from [here](https://drive.google.com/file/d/12sn9Mcr2HkITKLsXlHtjsE12lm7LhIxf/view?usp=sharing)*

-   `src/`
    -   `app/`
        -   `celery_app.py`
        -   `config.py`
        -   `main.py`
        -   `processing.py`

    -   `classic/`: &emsp;&emsp;*implementation of classic detection approaches*
        -   `dbscan.py`
        -   `dbscan_centroids.py`
        -   `morph_transforms.py`
    -   `detection/`: &emsp;*implementation of detection using Faster R-CNN and the final pipeline.*
        -   `constants.py`
        -   `EDA.py`
        -   `models.py`
        -   `train.py`: &emsp;*Training Faster R-CNN on the [ai-forever/school_notebooks_RU](https://huggingface.co/datasets/ai-forever/school_notebooks_RU) dataset.*
        -   `models.py`
-   `.dockerignore`
-   `.env`: &emsp;&emsp;&emsp; *environment used for broker configuration*
-   `.gitignore`
-   `app_requirements.txt`: &emsp; *requirements for Docker Image*
-   `compose.yaml`
-   `Dockerfile`: &emsp;*is used for both Celery and FastAPI*
-   `LICENSE.txt`
-   `README.md`
-   `requirements.txt`: &emsp;&emsp; *requirements for the entire project*

## Results Examples for Pipeline Stages
### Detected Word Clusters
This example demonstrates the pipeline's ability to detect objects using [Faster R-CNN](https://arxiv.org/pdf/1506.01497) and then cluster them using [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html). Different clusters are visualized with distinct colors.

<img src="data\results\clustering\5440470954456245678.jpg" alt="Example of Faster R-CNN detection and clustering with DBSCAN" width="100%"><br>


**Details:**

*   **Epsilon (eps):** 8
*   **Image Resolution:** 800x800

### Recognized Texts for Each Cluster

This example shows the result of [TrOCR](https://arxiv.org/abs/2109.10282) applied to each identified cluster. The text within each cluster is extracted and displayed.

<img src="data\results\recognition\5440470954456245678.jpg" alt="Example of recognised texts for each cluster" width="100%"><br>

## Install from Source Code

This is the recommended way to run OCR-based attendance tracking system. Docker Compose simplifies the process by managing all necessary services and dependencies.

## Prerequisites

Before you begin, ensure you have the following prerequisites met:

*   [Docker](https://www.docker.com/get-started) installed and running.

*   [CUDA 12.8 Drivers](https://developer.nvidia.com/cuda-downloads) installed.
<br>**Note:** The system requires CUDA to interact with the gpu. If the gpu has less memory than specified in [MIN_CUDA_MEMORY_REQUIRED](Dockerfile) (4 GB), the models will run on the CPU instead.

**Steps:**

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/isobreak/OCR-based-attendance-tracking-system.git
    ```
    ...and move to the project directory:
    ```bash
    cd OCR-based-attendance-tracking-system
    ```

2.  **Download the Models:**

    *   Download and unzip [*production*](https://drive.google.com/file/d/12sn9Mcr2HkITKLsXlHtjsE12lm7LhIxf/view?usp=sharing) folder into *data/models/*

3.  **Start the Application:**

    *   Run the following command to build and start all services defined in the `docker-compose.yml` file:

        ```bash
        docker-compose up -d
        ```

        *   The `-d` flag runs the containers in detached mode (in the background).  Remove `-d` to see the logs directly in your terminal.
        *   By default, the application will be accessible in your browser at http://localhost:8000.


## Usage Instructions

1. **Service Startup:**

   Ensure that the service is running and accessible at `http://127.0.0.1:8000`.

2. **Service Health Check:**

   Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to view the interactive Swagger UI documentation. Here you can explore the available API endpoints and test them.

3. **Request Format:**

   The service expects a POST request containing:

   *   **List of Images:** The image files to be processed.
   *   **JSON-encoded List of Names:** The `names` parameter should contain a JSON representation of a list of full names. Each full name should be represented as a list of strings.

       ```json
       [
           ["John", "Doe", "Smith"],
           ["Jane", "Marie", "Doe"],
           ["Peter", "Paul", "Jones"]
       ]
       ```
&emsp;&emsp;**Note:** List of Names **must** be json-encoded.

## Result Data Format

The `result` data returned by the `/task/{task_id}` endpoint is a JSON object containing the recognized names and metadata about the matched and unmatched clusters of bounding boxes.

```json
{
  "status": "completed",
  "result": {
    "ids": [],
    "scores": [],
    "meta": [
      {
        "image_number_in_group": 0,
        "matched": {
          "clusters": [],
          "scores": [],
          "names": []
        },
        "unmatched": {
          "clusters": []
        }
      }
    ]
  }
}
```

## ⚡ Inference Speed

These results reflect the typical performance of the model on the full dataset, without any optimizations applied. Future versions will use [ONNX Runtime](https://onnxruntime.ai/) for inference.
*   **Average Image Processing Time:** ~5 seconds
*   **Hardware Configuration:**
    *   **GPU:** RTX 4070 Super
    *   **VRAM:** 12 GB
    *   **CUDA Cores:** 7168
