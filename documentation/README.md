# Ray Verify
### AI-Assisted Screening of Patient-Acquired and Synthetic Chest X-ray Images

Ray Verify is a research prototype designed to help medical-imaging research data curators screen chest X-ray images by estimating whether an image is **Patient-acquired** or **Synthetic**. The project uses transfer learning with convolutional neural networks (CNNs), compares ResNet-18 and DenseNet-121, and applies an uncertainty-based withholding mechanism to avoid forcing a definitive prediction when the model score is below a selected threshold.

The deployed application uses a five-model DenseNet-121 ensemble and provides a Gradio interface for image analysis and curator review. Docker Compose is used to organize the application and monitoring services, with Prometheus collecting application metrics and Grafana displaying monitoring dashboards.

**Important:** Ray Verify is a research screening aid. It is not a medical diagnostic tool, an authenticity certificate, a fraud detector, or a replacement for source documentation and human review. A model prediction does not prove an image's true origin.

---

## 1. Project Motivation

Medical-imaging research datasets may contain images from different sources and acquisition processes. As generative AI becomes more capable, research data curators may need additional screening tools to identify images that appear synthetic before including them in a dataset.

Ray Verify explores whether supervised image classification can provide a useful first-pass estimate of image origin. It follows a curator-oriented workflow in which a user uploads an image, the model returns an estimated class and score, and uncertain cases are referred for further human review.

The project focuses on **image-origin screening**, rather than disease classification. It does not attempt to identify abnormalities, diagnose patients, or assess clinical image quality.

## 2. What the System Does

The prototype follows this workflow:

1. Accepts a chest X-ray image through the Gradio interface.
2. Converts the image to grayscale and applies the required preprocessing transformations.
3. Passes the image through five trained DenseNet-121 fold models.
4. Averages the class scores produced by the five models.
5. Applies a fixed withholding threshold of 0.80.
6. Returns one of three operational outcomes: Patient-acquired, Synthetic, or Uncertain.
7. Displays the prediction and average scores to the user.
8. Allows a curator to record an inclusion, exclusion, or unresolved decision with review notes for the current session.

The system also exposes application metrics for Prometheus, allowing prediction counts, inference duration, and related monitoring information to be visualized in Grafana.

## 3. Scope and Classification Labels

The implemented classification task contains two dataset ground-truth classes:

| Label | Meaning |
|---|---|
| 0 | Patient-acquired chest X-ray |
| 1 | Synthetic chest X-ray |

**Uncertain** is not a third ground-truth dataset class. It is an operational model outcome produced when the maximum average ensemble score is below 0.80.

The earlier proposed Mixed AI-edited category and secondary image attributes are outside the final implemented scope.

The system is intended to assist research dataset screening. It does not independently establish the provenance or authenticity of an image.

## 4. Datasets and Provenance

The experiments use two Kaggle sources:

| Collection | Source | Originally downloaded images |
|---|---|---:|
| Patient-acquired | [VinBigData Chest X-ray Resized PNG 256×256](https://www.kaggle.com/datasets/xhlulu/vinbigdata-chest-xray-resized-png-256x256) | 6,320 |
| Synthetic | [Synthetic Chest X-rays Kaggle notebook/input](https://www.kaggle.com/code/farihabintasadeque/synthetic-chestxrays/input?select=train) | 6,326 |
| **Total** | | **12,646** |

The patient-acquired resized collection is associated with the VinBigData Chest X-ray Abnormalities Detection competition. The original competition rules and the dataset page were reviewed. The educational-use-only restriction was identified in the original competition rules.

The synthetic Kaggle notebook describes a generation method, but the original image sources and explicit license or usage permission were not identified. Its provenance and permission status therefore remain unresolved. The project does not assume that the synthetic collection is licensed for unrestricted use or redistribution.

### Dataset Count Notes

The project records several counts from different pipeline stages:

- 12,646 originally downloaded images.
- 12,608 metadata records.
- 12,605 model inputs.

These counts are not interchangeable. The notebook uses different file-discovery rules for some inventory and model-input operations. Source-wise counts should be regenerated from the exact data folders used for a particular experiment before publishing a final reconciliation.

The datasets themselves are not included in this repository.

## 5. Machine Learning Models

Two pretrained CNN architectures were compared during model development.

### ResNet-18

ResNet-18 serves as the baseline architecture. It uses pretrained ImageNet feature layers and a replacement fully connected layer for the two-class image-origin classification task.

### DenseNet-121

DenseNet-121 is the comparison architecture and the selected architecture for the current deployed prototype. Five fold-specific models are used for ensemble inference.

### Ensemble Inference

For each uploaded image, the five DenseNet-121 models generate class scores. The system averages their softmax outputs to produce the final ensemble scores.

The average ensemble output is then evaluated using the withholding threshold:

- Maximum score ≥ 0.80: return the class with the higher average score.
- Maximum score < 0.80: return **Uncertain**.

The displayed scores are uncalibrated model scores. They should not be interpreted as probabilities that the prediction is correct.

## 6. Model Preprocessing and Training

The model input pipeline includes:

- Grayscale conversion.
- Conversion to three channels for pretrained ImageNet architectures.
- Torchvision pretrained-weight evaluation transformations.
- A model input shape of 3 × 224 × 224.
- A small brightness variation during training using `ColorJitter`.

Validation and test transformations are deterministic and do not apply random augmentation.

The model development workflow includes transfer learning, stratified splitting, five-fold cross-validation, model checkpoint saving, and independent test evaluation.

The same independent test manifest is used for the ResNet-18 and DenseNet-121 comparison.

## 7. Evaluation Methodology and Metrics

The reported independent test set contains 2,521 images:

| Class | Test images |
|---|---:|
| Patient-acquired | 1,264 |
| Synthetic | 1,257 |
| **Total** | **2,521** |

The evaluation includes the following metrics:

- **Precision:** Proportion of images predicted as a class that match that class's dataset label.
- **Recall:** Proportion of images belonging to a class that are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Coverage:** Proportion of images receiving a definitive class instead of Uncertain.
- **Error among classified:** Incorrect definitive predictions divided by all definitive predictions.
- **Uncertain count:** Number of images withheld from definitive classification.

The fixed 0.80 threshold is applied to the maximum average ensemble score. It is an operational withholding threshold, not a demonstrated calibration of model confidence.

DenseNet-121 was selected for the current prototype based on the reported comparison. The available experiment does not establish generalization to other hospitals, acquisition devices, patient populations, or unseen synthetic generators.

## 8. Explainability, Fairness, and Robustness

### LIME Explainability

LIME was applied to DenseNet-121 to generate a local, superpixel-based explanation for an individual image. The recorded experiment used 1,000 perturbation samples.

LIME highlights regions that influence the model's local prediction. It does not prove that the model relies on medically meaningful features or that the prediction is correct.

### Fairness Exploration with AIF360

AIF360 was explored, but a meaningful group-fairness audit could not be completed because the dataset does not contain independent protected attributes or demographic group information.

The image-origin label is the prediction target and must not be reused as a protected attribute for a group-fairness claim. Therefore, no valid group-fairness conclusion is reported.

### FGSM Adversarial Robustness

DenseNet-121 was stress-tested using the Fast Gradient Sign Method (FGSM) with epsilon = 8/255.

The recorded clean test accuracy was 98.73%, while accuracy on FGSM-perturbed images was 43.79%, a decrease of 54.94 percentage points.

This indicates substantial sensitivity to adversarial perturbations and is an important robustness limitation.

## 9. Repository Structure

The repository is organized into separate folders for deployment, documentation, monitoring, and source code.

```text
Ray-Verify-/
│
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── documentation/
│   └── README.md
│
├── monitering/
│   └──  prometheus/
│       └── prometheus.yml
│
├── src/
│   ├── app.py
│   ├── input_audit.csv
│   ├── playground.ipynb
│   ├── requirements.txt
│   ├── runs/
│   │   └── densenet121_10epochs_20261021_183511_055194/
│   │       └── ensemble_test_predictions.csv
│   ├── testing.ipynb
│   └── training.ipynb
│
├── training_experiments.ipynb
└── .gitignore
```

### Folder Descriptions

| Folder or file | Description |
|---|---|
| `deployment/` | Contains the Dockerfile and Docker Compose configuration for the application and monitoring services. |
| `documentation/` | Contains the project documentation and README. |
| `monitering/` | Contains the monitoring configuration. |
| `monitering/ prometheus/` | Contains the Prometheus configuration file. The folder currently has a leading space in its name. |
| `src/` | Contains the Gradio application, dependencies, notebooks, audit data, and experiment outputs. |
| `src/app.py` | Main deployed DenseNet-121 ensemble application. |
| `src/requirements.txt` | Python dependencies for the application. |
| `src/training.ipynb` | Model training notebook. |
| `src/testing.ipynb` | Model testing and evaluation notebook. |
| `src/playground.ipynb` | Supporting experimentation notebook. |
| `src/runs/` | Contains saved experiment outputs. |
| `training_experiments.ipynb` | Training and experiment notebook maintained at the repository root. |

The dataset files and trained ensemble checkpoint are not included in the visible repository tree. The application requires the appropriate model artifact at runtime.

## 10. Technologies Used

| Category | Technologies |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch, Torchvision |
| Architectures | ResNet-18, DenseNet-121 |
| Image processing | Pillow, Torchvision transforms |
| Interface | Gradio |
| Containerization | Docker, Docker Compose |
| Metrics collection | Prometheus |
| Dashboard and visualization | Grafana |
| Explainability | LIME |
| Fairness exploration | AI Fairness 360 |
| Robustness testing | FGSM |
| Experiment development | Jupyter Notebook |

## 11. Deployment Using Docker

Ray Verify uses Docker Compose to organize the application, Prometheus, and Grafana services.

### Prerequisites

- Docker Desktop or Docker Engine.
- Docker Compose.
- The Ray Verify repository.
- The DenseNet-121 ensemble checkpoint.
- The application dependencies specified in `src/requirements.txt`.

### Important Deployment Note

The current repository places the Dockerfile and Docker Compose file inside `deployment/`, while the Dockerfile expects `requirements.txt`, `app.py`, and `saved_models/` in its build context. The Compose file also uses relative paths.

Therefore, the deployment configuration must be checked against the actual build context and locations of the application files and model artifacts before the commands below can work. The current repository tree does not show a `saved_models/` folder or `densenet121_ensemble.pkl`.

### Accessing the Project Locally

Clone the repository:

```bash
git clone https://github.com/sarasanivedareddy/Ray-Verify-.git
cd Ray-Verify-
```

The intended Docker Compose startup command is:

```bash
docker compose -f deployment/docker-compose.yml up -d --build
```

Check running services:

```bash
docker compose -f deployment/docker-compose.yml ps
```

View application logs:

```bash
docker compose -f deployment/docker-compose.yml logs -f app
```

Stop the services:

```bash
docker compose -f deployment/docker-compose.yml down
```

These commands assume that the Compose build context, Dockerfile paths, and monitoring volume paths have been aligned with the repository's current folder structure.

## 12. Accessing the Services

When the containers start successfully, the services are intended to be available at:

| Service | Local URL | Purpose |
|---|---|---|
| Gradio Application | http://localhost:7860 | Upload and analyze chest X-ray images. |
| Prometheus | http://localhost:9090 | Query application metrics. |
| Grafana | http://localhost:3000 | View monitoring dashboards. |
| Application Metrics | http://localhost:8000/metrics | Exposes Prometheus metrics from the application. |

The Gradio application is configured with `share=False`, so a public Gradio share link is not enabled.

## 13. Monitoring with Prometheus

Prometheus collects application metrics exposed by the Gradio application.

The application starts a metrics server on port 8000 and exposes the following metrics:

| Metric | Description |
|---|---|
| `ray_verify_predictions_total` | Total number of successful predictions, with a `prediction` label for the outcome. |
| `ray_verify_prediction_duration_seconds` | Histogram measuring prediction processing duration. |
| `ray_verify_prediction_confidence` | Maximum average ensemble score from the latest prediction. |

Prometheus is configured to scrape the application metrics endpoint. The intended monitoring target is the application service at port 8000.

Example PromQL queries:

**Total predictions**

```promql
sum(ray_verify_predictions_total)
```

**Predictions by outcome**

```promql
sum by (prediction) (ray_verify_predictions_total)
```

**Average inference duration**

```promql
rate(ray_verify_prediction_duration_seconds_sum[5m])
/
rate(ray_verify_prediction_duration_seconds_count[5m])
```

**Latest prediction confidence**

```promql
ray_verify_prediction_confidence
```

Prometheus collects operational metrics. Prediction counts describe the model outputs produced by the running application; they do not measure accuracy unless ground-truth labels are independently available.

## 14. Monitoring with Grafana

Grafana provides dashboards for visualizing the metrics collected by Prometheus.

The Prometheus data source should use the internal Docker service address:

```text
http://prometheus:9090
```

The dashboard includes panels for:

- Total predictions.
- Average inference time.
- Synthetic prediction percentage.
- Memory usage.
- Prediction distribution.
- Application status.

The prediction distribution panel displays the number of Patient-acquired, Synthetic, and Uncertain outputs.

The application status panel uses Prometheus target availability to show whether the application metrics endpoint is reachable.

### Observed Monitoring Snapshot

During the documented monitoring session:

| Metric | Observed value |
|---|---:|
| Total predictions | 10 |
| Patient-acquired predictions | 4 |
| Synthetic predictions | 3 |
| Uncertain predictions | 3 |
| Average inference time | 0.695 seconds |
| Prometheus target status | 1/1 UP |

These values represent one observed application session, not a benchmark or guaranteed production performance.

## 15. Monitoring Workflow

The monitoring workflow is:

1. The user uploads an image through the Gradio interface.
2. The application preprocesses the image and performs DenseNet-121 ensemble inference.
3. The application updates prediction count, inference duration, and confidence metrics.
4. Prometheus scrapes the metrics endpoint.
5. Grafana queries Prometheus.
6. Dashboard panels visualize prediction distribution, latency, and application availability.

This setup provides basic operational visibility into the running application.

## 16. Known Limitations and Risks

1. **Synthetic source permissions:** The synthetic Kaggle notebook describes a generation method but does not provide explicit usage permission or complete original-source provenance.
2. **Patient-level leakage:** Patient and study identifiers are unavailable, so patient-level or study-level overlap between training and test images could not be verified. The split is image-level and stratified.
3. **Dataset representativeness:** The models were trained and evaluated on two specific Kaggle collections. Source-specific artifacts may influence predictions, and unseen generators may behave differently.
4. **Dataset count reconciliation:** Downloaded, metadata, and model-input counts come from different pipeline stages and file-discovery rules.
5. **Fairness:** Demographic and protected attributes are unavailable, preventing a meaningful group-fairness audit.
6. **Adversarial sensitivity:** FGSM testing showed a substantial reduction in accuracy.
7. **Score calibration:** The 0.80 threshold is a fixed operational threshold; the model scores have not been established as calibrated probabilities.
8. **External validation:** No independent external dataset from a separate institution or unseen generator is included in the reported comparison.
9. **Runtime limitations:** The application accepts one image at a time. Batch processing and concurrent request performance have not been established.
10. **Curator record persistence:** Curator decisions and notes are displayed for the current session only and are not stored in a database.
11. **Input validation:** The prototype handles unreadable images, but it does not verify whether an uploaded image is actually a chest X-ray.
12. **Deployment artifacts:** The ensemble checkpoint and datasets are not included in the visible repository structure and must be made available separately.

## 17. Responsible Use and Data Privacy

Ray Verify should only be used as an experimental screening aid by people who understand its limitations.

A model output must not be treated as proof that an image is genuine or synthetic. Curators should independently verify source documentation, applicable permissions, and provenance.

The application instructs users to upload only de-identified research images. Curator decisions and notes are not persisted to a database in the current implementation. The application does not provide a complete identity-management, access-control, or audit-trail system.

Do not use Ray Verify for diagnosis, treatment decisions, patient triage, clinical decision-making, or automated rejection of research images. Do not upload identifiable medical images to an unapproved hosted service. Follow data provider terms, institutional policies, and applicable privacy requirements.

## 18. Future Improvements

Potential future improvements include:

- Resolve and document the synthetic collection's original provenance and usage permission before broader use.
- Reconcile source-wise counts from original downloads through cleaning, metadata generation, and model input selection.
- Complete and verify an image-level lineage manifest with file hashes and split/fold membership.
- Obtain patient/study grouping metadata where permitted and evaluate group-based splitting to reduce related-image leakage risk.
- Evaluate on an external dataset and on synthetic images from generators not represented during training.
- Add calibration analysis and justify the withholding threshold using development data without tuning on the independent test set.
- Investigate stronger robustness methods and evaluate additional attacks.
- Measure inference latency, peak memory, and throughput on documented hardware.
- Conduct a meaningful fairness evaluation if appropriate independent group attributes become available.
- Add batch image upload and processing.
- Explore mixed AI-edited image identification as a separate classification category.
- Improve persistent curator feedback and review-history management.
- Expand monitoring with dedicated validation-failure metrics and alerting.

## 19. Acknowledgments and References

- xhlulu, [VinBigData Chest X-ray Resized PNG 256×256](https://www.kaggle.com/datasets/xhlulu/vinbigdata-chest-xray-resized-png-256x256).
- [VinBigData Chest X-ray Abnormalities Detection competition](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection).
- Fariha Binta Sadeque, [Synthetic Chest X-rays Kaggle notebook/input](https://www.kaggle.com/code/farihabintasadeque/synthetic-chestxrays/input?select=train).
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html).
- [Torchvision documentation](https://pytorch.org/vision/stable/index.html).
- [Gradio documentation](https://www.gradio.app/docs).
- [Docker documentation](https://docs.docker.com/).
- [Prometheus documentation](https://prometheus.io/docs/).
- [Grafana documentation](https://grafana.com/docs/).
- [LIME documentation](https://lime-ml.readthedocs.io/).
- [AI Fairness 360 documentation](https://aif360.readthedocs.io/).
- [Torchattacks documentation](https://adversarial-attacks-pytorch.readthedocs.io/).