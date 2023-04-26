import os
from flask import Flask, request, render_template

from src.model import CNN
from src.config import default_config
from src.s3_downloader import S3Downloader
from src.utils import get_df_infos, predict

# TODO: Add the aws access key and id inside a environment variable or using IAM roles

application = Flask(__name__)

@application.before_first_request
def startup():
    s3_object = S3Downloader(
        aws_access_key_id=default_config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=default_config.AWS_SECRET_ACCESS_KEY,
        region_name=default_config.AWS_ACCESS_REGION,
    )

    global model, disease_info, supplement_info

    model = CNN(num_classes=default_config.MODEL_NUM_CLASSES)
    model = s3_object.download_model(
        model=model,
        bucket_name=default_config.AWS_BUCKET_NAME,
        bucket_key=default_config.AWS_BUCKET_KEY,
        file_name_to_download=default_config.LOCAL_DOWNLOADED_MODEL_PATH,
    )
    model.eval()

    disease_info, supplement_info = get_df_infos(
        default_config.DISEASE_INFO_CSV_PATH, default_config.SUPPLEMENT_INFO_CSV_PATH
    )


@application.route("/")
def home_page():
    return render_template("home.html")


@application.route("/contact")
def contact():
    return render_template("contact-us.html")


@application.route("/index")
def ai_engine_page():
    return render_template("index.html")


@application.route("/mobile-device")
def mobile_device_detected_page():
    return render_template("mobile-device.html")


@application.route("/market", methods=["GET", "POST"])
def market():
    return render_template(
        "market.html",
        supplement_image=list(supplement_info["supplement image"]),
        supplement_name=list(supplement_info["supplement name"]),
        disease=list(disease_info["disease_name"]),
        buy=list(supplement_info["buy link"]),
    )


@application.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        image_request = request.files["image"]
        image_file_name = image_request.filename
        image_save_path = os.path.join(
            default_config.IMAGE_UPLOAD_PATH, image_file_name
        )
        image_request.save(image_save_path)
        prediction_json_response = predict(
            model = model, 
            image_path = image_save_path, 
            disease_info = disease_info, supplement_info = supplement_info
        )
        
        return render_template(
            'submit.html' , 
            title = prediction_json_response['title'], 
            desc = prediction_json_response['description'], 
            prevent = prediction_json_response['prevent'], 
            image_url = prediction_json_response['image_url'] , 
            pred = prediction_json_response['pred'], 
            sname = prediction_json_response['supplement_name'], 
            simage = prediction_json_response['supplement_image_url'], 
            buy_link = prediction_json_response['supplement_buy_link']
        )


if __name__ == "__main__":
    application.run(port=8000)
