# S3 Storage Configuration

## Overview

This document provides the S3 bucket configuration for storing audio, image, and text data for the Dyslexia Early Detection System.

## Bucket Structure

```
dyslexia-detection-prod/
├── raw/                          # Original, unmodified data
│   ├── speech/                   # Raw audio files
│   │   ├── uploads/              # User uploads
│   │   └── validated/             # QA passed
│   ├── handwriting/              # Raw handwriting images
│   │   ├── uploads/
│   │   └── validated/
│   └── text/                     # Raw text files
│       ├── uploads/
│       └── validated/
│
├── processed/                    # Preprocessed data
│   ├── speech/                   # Processed audio
│   ├── handwriting/               # Processed images
│   └── text/                     # Processed text
│
├── models/                       # Trained ML models
│   ├── speech/
│   ├── handwriting/
│   └── text/
│
├── features/                     # Extracted features
│   ├── speech/
│   ├── handwriting/
│   └── text/
│
├── exports/                      # Data exports
│   ├── reports/
│   └── research/
│
└── temp/                         # Temporary processing
    └── jobs/
```

## Terraform Configuration

```hcl
resource "aws_s3_bucket" "dyslexia_data" {
  bucket = "dyslexia-detection-prod"
  
  tags = {
    Project     = "Dyslexia Detection System"
    Environment = "production"
    DataType    = "sensitive"
  }
}

resource "aws_s3_bucket_versioning" "dyslexia_data" {
  bucket = aws_s3_bucket.dyslexia_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dyslexia_data" {
  bucket = aws_s3_bucket.dyslexia_data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "dyslexia_data" {
  bucket = aws_s3_bucket.dyslexia_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policy
resource "aws_s3_bucket_lifecycle_configuration" "dyslexia_data" {
  bucket = aws_s3_bucket.dyslexia_data.id
  
  rule {
    id     = "move-to-glacier"
    status = "Enabled"
    
    transition {
      days          = 365
      storage_class = "GLACIER"
    }
    
    transition {
      days          = 1825  # 5 years
      storage_class = "DEEP_ARCHIVE"
    }
    
    expiration {
      days = 2555  # 7 years
    }
  }
  
  rule {
    id     = "delete-temp"
    status = "Enabled"
    
    expiration {
      days = 7
    }
    
    filter {
      prefix = "temp/"
    }
  }
}
```

## IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListAllBuckets",
      "Effect": "Allow",
      "Action": ["s3:ListAllBuckets"],
      "Resource": "*"
    },
    {
      "Sid": "ListAndGetObjects",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:GetObjectVersion"
      ],
      "Resource": "arn:aws:s3:::dyslexia-detection-prod/*"
    },
    {
      "Sid": "PutObjects",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl"
      ],
      "Resource": "arn:aws:s3:::dyslexia-detection-prod/raw/*"
    },
    {
      "Sid": "DeleteObjects",
      "Effect": "Allow",
      "Action": [
        "s3:DeleteObject",
        "s3:DeleteObjectVersion"
      ],
      "Resource": "arn:aws:s3:::dyslexia-detection-prod/temp/*"
    }
  ]
}
```

## Access Patterns

### Application Access (Production)

```python
import boto3
import os

class S3Storage:
    def __init__(self):
        self.bucket = os.environ['S3_BUCKET']
        self.client = boto3.client('s3')
    
    def upload_file(self, file_path, key):
        self.client.upload_file(
            file_path,
            self.bucket,
            key,
            ExtraArgs={
                'ServerSideEncryption': 'AES256',
                'ContentType': self._get_content_type(key)
            }
        )
    
    def download_file(self, key, local_path):
        self.client.download_file(self.bucket, key, local_path)
    
    def get_presigned_url(self, key, expiration=3600):
        return self.client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expiration
        )
    
    def _get_content_type(self, key):
        ext = key.split('.')[-1].lower()
        types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'txt': 'text/plain',
            'json': 'application/json'
        }
        return types.get(ext, 'application/octet-stream')
```

## Cross-Region Replication (Optional)

For disaster recovery, enable CRR:

```hcl
resource "aws_s3_bucket_replication_configuration" "dyslexia_data" {
  role = aws_iam_role.replication.arn
  
  rule {
    id     = "replicate-to-dr"
    status = "Enabled"
    
    destination {
      bucket        = "dyslexia-detection-dr"
      storage_class = "STANDARD"
    }
  }
}
```

## Monitoring

### CloudWatch Metrics
- `Requests` - Total requests
- `BytesUploaded` - Upload volume
- `BytesDownloaded` - Download volume
- `Errors` - Error count

### Event Notifications
```hcl
resource "aws_s3_bucket_notification" "dyslexia_data" {
  bucket = aws_s3_bucket.dyslexia_data.id
  
  lambda_function {
    lambda_function_arn = aws_lambda_function.processor.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/"
  }
}
```

---

*Configuration Version: 1.0*  
*Last Updated: 2026-04-12*