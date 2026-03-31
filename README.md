# Spark ML Pipeline

A scalable machine learning pipeline built with Apache Spark for distributed data processing and model training.

## Features
- **Distributed Processing:** Leverages Spark's distributed computing capabilities for large datasets.
- **MLlib Integration:** Utilizes Spark MLlib for various machine learning algorithms and utilities.
- **Data Ingestion & Preprocessing:** Includes modules for efficient data loading, cleaning, and transformation.
- **Model Training & Evaluation:** Supports training, hyperparameter tuning, and evaluation of ML models.
- **Deployment Ready:** Designed for easy deployment on Spark clusters (YARN, Mesos, Kubernetes).

## Getting Started

### Prerequisites
- Apache Spark (3.0+)
- Python 3.8+
- Java 8+

### Installation

```bash
git clone https://github.com/Dras1950/spark-ml-pipeline.git
cd spark-ml-pipeline
pip install -r requirements.txt
```

### Usage

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Initialize Spark Session
spark = SparkSession.builder.appName("SparkMLPipeline").getOrCreate()

# Sample Data
data = [
    (0.0, 0.5, 0.2, 0.7, 0.0),
    (0.1, 0.6, 0.3, 0.8, 0.0),
    (0.2, 0.7, 0.4, 0.9, 0.0),
    (1.0, 0.3, 0.8, 0.1, 1.0),
    (1.1, 0.4, 0.9, 0.2, 1.0),
    (1.2, 0.5, 1.0, 0.3, 1.0)
]
columns = ["feature1", "feature2", "feature3", "feature4", "label"]
df = spark.createDataFrame(data, columns)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=columns[:-1], outputCol="features")

# Logistic Regression Model
lr = LogisticRegression(labelCol="label", featuresCol="features")

# Create Pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Train Model
model = pipeline.fit(df)

# Make predictions
predictions = model.transform(df)
predictions.select("features", "label", "prediction").show()

spark.stop()
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
