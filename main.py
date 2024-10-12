import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO


# تحميل النموذج الخاص بك من المسار المحدد
model_path = "../Yolo-Weights/auto.pt"  # قم بتحديد مسار النموذج الخاص بك
model = YOLO(model_path)

def yolov10_inference(image, video, image_size, conf_threshold):
    if image:
        # معالجة الصورة باستخدام النموذج الخاص بك
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    else:
        # معالجة الفيديو باستخدام النموذج الخاص بك
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolov10_inference_for_examples(image, image_size, conf_threshold):
    annotated_image, _ = yolov10_inference(image, None, image_size, conf_threshold)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                # إزالة خيار تحديد النموذج، لأنه ثابت الآن
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image_update = gr.update(visible=input_type == "Image")
            video_update = gr.update(visible=input_type == "Video")
            output_image_update = gr.update(visible=input_type == "Image")
            output_video_update = gr.update(visible=input_type == "Video")

            return image_update, video_update, output_image_update, output_video_update

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov10_inference(image, None, image_size, conf_threshold)
            else:
                return yolov10_inference(None, video, image_size, conf_threshold)


        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    640,
                    0.25,
                ],
            ],
            fn=yolov10_inference_for_examples,
            inputs=[
                image,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Object Detection with Your Custom Model
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch()
