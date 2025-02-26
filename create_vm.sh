# gcloud compute instances create "unuu-assign" \
#   --zone="asia-northeast1-c" \
#   --image-family="pytorch-latest-cu113" \
#   --machine-type="g2-standard-8" \
#   --image-project=deeplearning-platform-release \
#   --maintenance-policy=TERMINATE \
#   --accelerator="type=nvidia-l4,count=1" \
#   --metadata="install-nvidia-driver=True" \
#   --boot-disk-size="200GB"
 
gcloud compute instances create "unuu-assign" \
  --zone="asia-northeast1-c" \
  --image-family="pytorch-latest-cu113" \
  --machine-type="n1-standard-8" \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True" \
  --boot-disk-size="200GB"

# Only available config I found.
gcloud compute instances create "unuu-assign" \
  --zone="asia-central1-c" \
  --image-family="pytorch-latest-cu113" \
  --machine-type="n1-standard-8" \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-p100,count=1" \
  --metadata="install-nvidia-driver=True" \
  --boot-disk-size="200GB"
