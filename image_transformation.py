import io
from PIL import Image

def get_image(image_file, max_size = 1024):
    image_input = Image.open(io.BytesIO(image_file)).convert("RGB")
    width, height = image_input.size
    image_resized = image_input.resize(
        (
            int(image_input.width * min(max_size / width, max_size / height)),
            int(image_input.height * min(max_size / width, max_size / height)),
        )
    )
    return image_resized

def get_image_output_transformation(results):
    results_rendered = results.render()
    for img_i in results_rendered:
        image_bytes = io.BytesIO()
        image_output_rendered = Image.fromarray(img_i)
        image_output = image_output_rendered.save(image_bytes, format="jpeg")
    return image_output, image_bytes
