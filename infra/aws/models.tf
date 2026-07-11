# Model-artefact bucket — AWS analog of gs://fantasy-coach-lcd-models.
# The precompute/retrain jobs upload the trained joblib here; the API + jobs
# download it on cold start (see fantasy_coach.object_store). Versioned so a
# bad promote can be rolled back to the prior object version.

resource "aws_s3_bucket" "models" {
  bucket = "${var.name_prefix}-models-${var.account_id}"
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Runtime read access (Lambda/Fargate task roles) is granted in Phase 2 when
# those consumers exist — no speculative principals here.
