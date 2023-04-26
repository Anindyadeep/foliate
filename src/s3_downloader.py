import boto3
import torch
from tqdm import tqdm

# aws_access_key_id='AKIATFLRHDCN3KRSS7CQ'
# aws_secret_access_key='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'
# region_name='ap-south-1'

# bucket_name = 'rootskart-users'
# key = 'plant-disease-model/plant_disease_model_1.pt'
# file_name = "model_from_s3.pt"


class S3Downloader:
    def __init__(
        self, aws_access_key_id: str, aws_secret_access_key: str, region_name: str
    ) -> None:
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        self.progress = None

    def download_progress(self, bytes_amount):
        self.progress.update(bytes_amount)

    def download_model(
        self, model: torch.nn.Module, bucket_name: str, bucket_key: str, file_name_to_download: str
    ) -> torch.nn.Module:
        s3_object = self.s3.head_object(Bucket=bucket_name, Key=bucket_key)

        file_size = s3_object["ContentLength"]
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Downloading", leave=True
        ) as self.progress:
            self.s3.download_file(
                bucket_name, bucket_key, file_name_to_download, Callback=self.download_progress
            )

        # either return the the file or the file_path
        print(f"=> Downloaded file as {file_name_to_download}")

        model_state_dict = torch.load(file_name_to_download, map_location="cpu")
        model.load_state_dict(model_state_dict)
        return model


if __name__ == "__main__":
    aws_access_key_id = "AKIATFLRHDCN3KRSS7CQ"
    aws_secret_access_key = "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"
    region_name = "ap-south-1"

    s3_downloader = S3Downloader(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

    s3_downloader.download_model(
        bucket_name="rootskart-users",
        bucket_key="plant-disease-model/plant_disease_model_1.pt",
        file_name_to_download="model_from_s3.pt",
    )
